# pyright: basic
"""Capture the Muse-Glimmer DFlash drafter's fp32 activations from transformers.

The drafter is not a language model: it takes the target's hidden states at
`target_layer_ids` (concatenated on the last dim) plus the raw embedding rows of
`[anchor, mask x 15]`, and returns 16 hidden rows the target's own `lm_head` turns into
logits. So the fixture's inputs are two tensors, not ids: the context features are drawn
from a seeded generator (the drafter's arithmetic does not care where they came from),
and the noise embeddings are the real lookup rows, read straight out of the target's
`embed_tokens` — the one place the two checkpoints touch, and the one worth pinning.

The noise floor is the same fp32 graph replayed in fp64, per tensor. Five layers is a
short trunk, but the residual still grows, so floors are per-block.

Sliding is not exercised here: the window is 2048 and a context that long is a 273 MB
input for one number. The window belongs in a synthetic-config probe, like the trunk's.

Run: MLX_ENABLE_TF32=0 uv run --with git+https://github.com/huggingface/transformers \
     --with torch --with safetensors --no-project python \
     packages/engine/tests/fixtures/generate_muse_glimmer_assistant.py
After regenerating, update SHA256SUMS.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pathlib

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoModel

HUB = pathlib.Path.home() / ".cache/huggingface/hub"
DRAFTER = "meta-models/Muse-Glimmer-30B-assistant"
TARGET = "meta-models/Muse-Glimmer-30B"
EMBED = "model.language_model.embed_tokens.weight"

CONTEXT = 40
"""Accepted positions the drafter conditions on — long enough to be a sequence, short
enough that fp64 stays cheap."""

ANCHOR = 3838
"""The last committed token, which is the block's first row. Any real id does; this one
is not special and the fixture pins it so the embedding lookup is reproducible."""


def snapshot(repository: str) -> pathlib.Path:
    directory = HUB / f"models--{repository.replace('/', '--')}" / "snapshots"
    return next(directory.iterdir())


def embedding_rows(ids: list[int]) -> torch.Tensor:
    """`embed_tokens` rows, raw — the drafter takes the lookup and not the target's
    `embed()`, which norms it (candidate_generator.py says so in as many words)."""
    directory = snapshot(TARGET)
    index = directory / "model.safetensors.index.json"
    import json

    shard = json.loads(index.read_text())["weight_map"][EMBED]
    with safe_open(directory / shard, framework="pt") as shards:
        table = shards.get_slice(EMBED)
        return torch.stack([torch.tensor(table[i : i + 1][0]) for i in ids]).float()


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def forward(model, context: torch.Tensor, noise: torch.Tensor) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    handles = [
        model.encoder.register_forward_hook(capture_into(captured, "encoder")),
        model.norm.register_forward_hook(capture_into(captured, "norm")),
    ]
    handles += [
        block.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, block in enumerate(model.layers)
    ]
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        output = model(
            noise_embeds=noise.to(dtype),
            context_hidden_states=context.to(dtype),
            use_cache=False,
        )
    for handle in handles:
        handle.remove()
    captured["last_hidden_state"] = output.last_hidden_state.detach().double().numpy()
    return captured


def main() -> None:
    model = AutoModel.from_pretrained(snapshot(DRAFTER), dtype=torch.float32)
    model.eval()
    config = model.config
    block, taps = config.block_size, len(config.target_layer_ids)

    generator = torch.Generator().manual_seed(0)
    context = torch.randn(
        1, CONTEXT, taps * config.hidden_size, generator=generator, dtype=torch.float32
    )
    ids = [ANCHOR] + [config.mask_token_id] * (block - 1)
    noise = embedding_rows(ids)[None]

    exact = forward(model, context, noise)
    model.double()
    reference = forward(model, context, noise)

    tensors: dict[str, np.ndarray] = {
        "context_hidden_states": context.numpy(),
        "noise_embeds": noise.numpy().astype(np.float32),
        "noise_ids": np.array(ids, dtype=np.int32),
    }
    for name, value in exact.items():
        tensors[name] = value.astype(np.float32)
        tensors[f"noise.{name}"] = np.array(
            [relative_diff(value, reference[name])], dtype=np.float32
        )

    path = pathlib.Path(__file__).parent / "muse_glimmer_assistant_forward.safetensors"
    save_file(tensors, path)
    print(f"{path}")
    for name in exact:
        print(f"  noise.{name}: {tensors[f'noise.{name}'][0]:.3e}")


if __name__ == "__main__":
    main()
