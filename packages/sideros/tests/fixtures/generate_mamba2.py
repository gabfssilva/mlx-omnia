# pyright: basic
"""Capture Mamba2 fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64, per tensor: the
residual stream grows along the blocks, so one tolerance across the trunk
would be vacuous at one end and impossible at the other.

Uses the naive `torch_forward` (the fast path needs CUDA-only mamba-ssm /
causal-conv1d). The reference checkpoint must be the HF transformers-format
variant (`model_type=mamba2`), not the original state-spaces format.

Run: uv run --with torch --with transformers --with safetensors \
     python packages/sideros/tests/fixtures/generate_mamba2.py
After regenerating, update SHA256SUMS (shasum -a 256 mamba2_forward.safetensors).
"""

import pathlib

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import Mamba2ForCausalLM

REPO = "state-spaces/mamba2-2.8b-hf"
IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(ids: list[int], dtype: torch.dtype) -> dict[str, np.ndarray]:
    model = Mamba2ForCausalLM.from_pretrained(REPO, dtype=dtype)
    model.eval()
    captured: dict[str, np.ndarray] = {}

    backbone = model.backbone
    block0 = backbone.layers[0]
    mixer0 = block0.mixer
    handles = [
        backbone.embeddings.register_forward_hook(capture_into(captured, "embeddings")),
        backbone.norm_f.register_forward_hook(capture_into(captured, "norm")),
        block0.norm.register_forward_hook(capture_into(captured, "b0_ln")),
        mixer0.out_proj.register_forward_hook(capture_into(captured, "b0_mixer")),
    ]
    handles += [
        b.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, b in enumerate(backbone.layers)
    ]

    with torch.no_grad():
        captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()

    for handle in handles:
        handle.remove()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = Mamba2ForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]), max_new_tokens=20, do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    return out[0].numpy().astype(np.int32)


def main() -> None:
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

    path = pathlib.Path(__file__).parent / "mamba2_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    n_layer = len([k for k in out if k.startswith("block_") and k[6:].isdigit()])
    print(f"{path}: {len(out)} arrays, {n_layer} layers")
    for name in ("block_0", f"block_{n_layer - 1}", "norm", "logits"):
        print(f"  {name:10s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
