# pyright: basic
"""Capture Gemma 3 270M fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64, per tensor: Gemma's residual
grows along the trunk, so one flat tolerance would be loose somewhere and impossible
elsewhere. Five of every six layers only attend within a 512-key window, so a short
prompt exercises the sliding and the full mask identically — hence the 600-token
sequence, whose blocks 0 (sliding) and 5 (full) are pinned as well.

Run: uv run --with torch --with transformers --with safetensors \
     python packages/engine/tests/fixtures/generate_gemma3.py
After regenerating, update SHA256SUMS.gemma3 (shasum -a 256 gemma3_forward.safetensors).
"""

import gc
import pathlib

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoTokenizer, Gemma3ForCausalLM

MODEL = "google/gemma-3-270m"
PROMPT = "The capital of France is"
LONG_LENGTH = 600

# Block 0 slides, block 5 attends fully: one internals set per layer type.
INTERNALS = (
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "mlp",
    "post_feedforward_layernorm",
)


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(model, ids: list[int], internals: bool, last_logit_only: bool = False):
    """Every tensor a parity test pins, in fp64 numpy regardless of the graph dtype."""
    captured: dict[str, np.ndarray] = {}
    handles = [
        model.model.embed_tokens.register_forward_hook(capture_into(captured, "embeddings")),
        model.model.norm.register_forward_hook(capture_into(captured, "norm")),
    ]
    handles += [
        b.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, b in enumerate(model.model.layers)
    ]
    if internals:
        for index in (0, 5):
            block = model.model.layers[index]
            handles += [
                getattr(block, name).register_forward_hook(
                    capture_into(captured, f"b{index}_{name}")
                )
                for name in INTERNALS
            ]
            handles += [
                getattr(block.self_attn, name).register_forward_hook(
                    capture_into(captured, f"b{index}_{name}")
                )
                for name in ("q_norm", "k_norm")
            ]

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
    # 600 x 262144 doubles would be a 1.2 GB fixture; only the last row drives generation.
    captured["logits"] = (logits[:, -1:, :] if last_logit_only else logits).double().numpy()

    for handle in handles:
        handle.remove()
    return captured


def pass_at(dtype: torch.dtype, ids: list[int], long_ids: list[int]) -> dict[str, np.ndarray]:
    model = Gemma3ForCausalLM.from_pretrained(MODEL, dtype=dtype).eval()
    captured = forward(model, ids, internals=True)
    long = forward(model, long_ids, internals=False, last_logit_only=True)
    captured["long_block_0"] = long["block_0"]
    captured["long_block_5"] = long["block_5"]
    captured["long_logits_last"] = long["logits"]
    del model, long
    gc.collect()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = Gemma3ForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    del model
    gc.collect()
    return out[0].numpy().astype(np.int32)


def main() -> None:
    ids = AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"]
    long_ids = (ids * (LONG_LENGTH // len(ids) + 1))[:LONG_LENGTH]

    captured = pass_at(torch.float32, ids, long_ids)
    exact = pass_at(torch.float64, ids, long_ids)

    out: dict[str, np.ndarray] = {
        name: tensor.astype(np.float32) for name, tensor in captured.items()
    }
    out.update(
        {
            f"noise.{name}": np.array([relative_diff(captured[name], tensor)], dtype=np.float32)
            for name, tensor in exact.items()
        }
    )
    out["input_ids"] = np.array(ids, dtype=np.int32)
    out["long_input_ids"] = np.array(long_ids, dtype=np.int32)
    out["greedy_ids"] = greedy_ids(ids)

    path = pathlib.Path(__file__).parent / "gemma3_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}; long sequence {LONG_LENGTH} tokens")
    for name in ("block_0", "block_5", "block_17", "norm", "logits", "long_block_0",
                 "long_block_5", "long_logits_last"):
        print(f"  {name:18s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
