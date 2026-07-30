# pyright: basic
"""Capture Gemma 4 E2B fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64, per tensor. The 35-layer
trunk grows its residual, so floors are per-block. The 600-token sequence separates
the 4:1 sliding/full split (full at idx 4,9,...) — below the 512-key window both
masks agree.

Gemma 4 specifics captured: PLE per-layer inputs, KV-sharing shared KV tensors,
proportional RoPE on full layers, logit softcap (fp32), standard (non-zero-centered)
RMSNorm, dual head_dim, scale=1.0.

Run: uv run --with torch --with transformers --with safetensors \
     python packages/sideros/tests/fixtures/generate_gemma4.py
After regenerating, update SHA256SUMS (shasum -a 256 gemma4_forward.safetensors).
"""

import gc
import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pathlib

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-4-e2b-it"
PROMPT = "The capital of France is"
LONG_LENGTH = 600

# Block 0 slides, block 4 attends fully (4:1 split, full at idx 4).
SLIDING_LAYER, FULL_LAYER = 0, 4

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
        for index in (SLIDING_LAYER, FULL_LAYER):
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
            # PLE arm internals (if present)
            if hasattr(block, "ple_arm"):
                handles += [
                    block.ple_arm.register_forward_hook(
                        capture_into(captured, f"b{index}_ple_arm")
                    )
                ]
            # Capture per-layer input
            if hasattr(model.model, "embed_tokens_per_layer"):
                pass  # per-layer input is computed in forward, not a submodule

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
    captured["logits"] = (
        logits[:, -1:, :] if last_logit_only else logits
    ).double().numpy()

    for handle in handles:
        handle.remove()
    return captured


def pass_at(dtype: torch.dtype, ids: list[int], long_ids: list[int]) -> dict[str, np.ndarray]:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval()
    captured = forward(model, ids, internals=True)
    long = forward(model, long_ids, internals=False, last_logit_only=True)
    captured["long_block_0"] = long["block_0"]
    captured[f"long_block_{FULL_LAYER}"] = long[f"block_{FULL_LAYER}"]
    captured["long_logits_last"] = long["logits"]
    del model, long
    gc.collect()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
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

    path = pathlib.Path(__file__).parent / "gemma4_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}; long sequence {LONG_LENGTH} tokens")
    for name in (
        "block_0", f"block_{FULL_LAYER}", "block_34", "norm", "logits",
        "long_block_0", f"long_block_{FULL_LAYER}", "long_logits_last",
    ):
        if name in out:
            print(f"  {name:18s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
