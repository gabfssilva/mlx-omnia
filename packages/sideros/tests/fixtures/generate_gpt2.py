# pyright: basic
"""Capture GPT-2 fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64, per tensor: the residual
stream grows along the 12 blocks, so one tolerance across the trunk would be vacuous
at one end and impossible at the other.

Run: uv run --with torch --with transformers --with safetensors \
     python packages/sideros/tests/fixtures/generate_gpt2.py
After regenerating, update SHA256SUMS (shasum -a 256 gpt2_forward.safetensors).
"""

import pathlib

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import GPT2LMHeadModel

IDS = [15496, 11, 616, 1438, 318]  # "Hello, my name is"


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(ids: list[int], dtype: torch.dtype) -> dict[str, np.ndarray]:
    """Every tensor a parity test pins, in fp64 numpy regardless of the graph dtype."""
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=dtype)
    model.eval()
    captured: dict[str, np.ndarray] = {}

    # Hooks pin each tensor to a module boundary; output_hidden_states is ambiguous
    # about whether the last entry is before or after ln_f.
    block0 = model.transformer.h[0]
    handles = [
        model.transformer.drop.register_forward_hook(capture_into(captured, "embeddings")),
        model.transformer.ln_f.register_forward_hook(capture_into(captured, "ln_f")),
        # Block 0 internals, to localize a divergence inside the block.
        block0.ln_1.register_forward_hook(capture_into(captured, "b0_ln_1")),
        block0.attn.register_forward_hook(capture_into(captured, "b0_attn")),
        block0.ln_2.register_forward_hook(capture_into(captured, "b0_ln_2")),
        block0.mlp.register_forward_hook(capture_into(captured, "b0_mlp")),
    ]
    handles += [
        b.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, b in enumerate(model.transformer.h)
    ]

    with torch.no_grad():
        captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()

    for handle in handles:
        handle.remove()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]), max_new_tokens=20, do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    return out[0].numpy().astype(np.int32)


def main() -> None:
    # The checkpoint is fp32 natively; fp64 upcasts it losslessly.
    exact = forward(IDS, torch.float64)
    captured = forward(IDS, torch.float32)

    out: dict[str, np.ndarray] = {
        name: tensor.astype(np.float32) for name, tensor in captured.items()
    }
    out.update(
        {
            f"noise.{name}": np.array([relative_diff(captured[name], tensor)], dtype=np.float32)
            for name, tensor in exact.items()
        }
    )
    out["input_ids"] = np.array(IDS, dtype=np.int32)
    out["greedy_ids"] = greedy_ids(IDS)

    path = pathlib.Path(__file__).parent / "gpt2_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    for name in ("block_0", "block_11", "ln_f", "logits"):
        print(f"  {name:10s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
