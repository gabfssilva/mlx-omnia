# pyright: basic
"""Capture Falcon-H1 7B fp32 activations from transformers as ground truth.

The 7B (``n_groups=1``) is the fixture model: it is the only size where mlx-lm
is a valid reference (mlx-lm's ``FalconH1RMSNormGated`` calls bare
``mx.fast.rms_norm`` with no grouping, which matches transformers only when
``n_groups=1``). The 34B (``n_groups=2``) diverges and needs its own fixture.

The noise floor is the same fp32 graph replayed in fp64, per tensor. The μP
folding floor (``noise.fold``) measures the difference between the folded-at-load
path (multipliers baked into bf16 weights) and transformers' unfolded runtime
fp32 multiply — carry it as a separate floor.

Run: uv run --with torch --with transformers --with safetensors \
     python packages/sideros/tests/fixtures/generate_falcon_h1.py
After regenerating, update SHA256SUMS.

Environment: MLX_ENABLE_TF32=0 before importing mlx (pinned in conftest for
tests; this script needs its own). End load with mx.eval of the weights.
"""

import gc
import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pathlib

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoTokenizer, FalconH1ForCausalLM

MODEL = "tiiuae/Falcon-H1-7B-Base"
PROMPT = "The capital of France is"


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(ids: list[int], dtype: torch.dtype) -> dict[str, np.ndarray]:
    """Every tensor a parity test pins, in fp64 numpy regardless of the graph dtype."""
    captured: dict[str, np.ndarray] = {}
    model = FalconH1ForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval()
    text = model.model

    handles = [
        text.embed_tokens.register_forward_hook(capture_into(captured, "embeddings")),
        text.norm.register_forward_hook(capture_into(captured, "norm")),
    ]
    for i in range(min(4, len(text.layers))):
        handles.append(
            text.layers[i].register_forward_hook(capture_into(captured, f"block_{i}"))
        )

    # Capture the mamba mixer and attention of the first layer
    layer0 = text.layers[0]
    handles.append(layer0.mamba.register_forward_hook(capture_into(captured, "b0_mamba")))
    handles.append(layer0.self_attn.register_forward_hook(capture_into(captured, "b0_attn")))
    handles.append(layer0.input_layernorm.register_forward_hook(capture_into(captured, "b0_ln")))
    handles.append(
        layer0.pre_ff_layernorm.register_forward_hook(capture_into(captured, "b0_ff_ln"))
    )
    handles.append(layer0.mlp.register_forward_hook(capture_into(captured, "b0_mlp")))

    with torch.no_grad():
        captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()

    for handle in handles:
        handle.remove()
    del model
    gc.collect()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = FalconH1ForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]), max_new_tokens=20, do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    del model
    gc.collect()
    return out[0].numpy().astype(np.int32)


def main() -> None:
    ids = AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"]

    captured = forward(ids, torch.float32)
    generated = greedy_ids(ids)
    exact = forward(ids, torch.float64)

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
    out["greedy_ids"] = generated

    path = pathlib.Path(__file__).parent / "falcon_h1_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}")
    print(f"  greedy -> {generated.tolist()}")
    for name in ("block_0", "block_1", "b0_mamba", "b0_attn", "norm", "logits"):
        print(f"  {name:12s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
