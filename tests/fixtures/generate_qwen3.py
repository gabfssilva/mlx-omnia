# pyright: basic
"""Capture Qwen3-0.6B fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64: at 28 layers the residual
grows along the trunk, so the floor is per tensor, never a single number.

Run: uv run --with torch --with transformers --with safetensors \
     python packages/engine/tests/fixtures/generate_qwen3.py
After regenerating, update SHA256SUMS (shasum -a 256 qwen3_forward.safetensors).
"""

import pathlib

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL = "Qwen/Qwen3-0.6B"
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
    model = Qwen3ForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval()
    captured: dict[str, np.ndarray] = {}

    block0 = model.model.layers[0]
    handles = [
        model.model.embed_tokens.register_forward_hook(capture_into(captured, "embeddings")),
        model.model.norm.register_forward_hook(capture_into(captured, "norm")),
        block0.input_layernorm.register_forward_hook(capture_into(captured, "b0_ln_1")),
        block0.self_attn.register_forward_hook(capture_into(captured, "b0_attn")),
        block0.post_attention_layernorm.register_forward_hook(capture_into(captured, "b0_ln_2")),
        block0.mlp.register_forward_hook(capture_into(captured, "b0_mlp")),
        # q_norm/k_norm sit between the projections and the rotation; their outputs pin
        # everything up to rope, and the rotary replay below pins rope itself.
        block0.self_attn.q_norm.register_forward_hook(capture_into(captured, "b0_q_norm")),
        block0.self_attn.k_norm.register_forward_hook(capture_into(captured, "b0_k_norm")),
    ]
    handles += [
        b.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, b in enumerate(model.model.layers)
    ]

    with torch.no_grad():
        captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()

    # Replay HF's own rotary application on the captured normed projections: same ground
    # truth, reachable from a boundary that does not exist as a module.
    q = torch.tensor(captured["b0_q_norm"]).transpose(1, 2).to(dtype)
    k = torch.tensor(captured["b0_k_norm"]).transpose(1, 2).to(dtype)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(q, torch.arange(len(ids))[None])
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    captured["b0_q_rope"] = q_rope.double().numpy()
    captured["b0_k_rope"] = k_rope.double().numpy()

    for handle in handles:
        handle.remove()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = Qwen3ForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]), max_new_tokens=20, do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    return out[0].numpy().astype(np.int32)


def main() -> None:
    ids = AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"]

    # The checkpoint is bfloat16; float32 upcasts it losslessly.
    exact = forward(ids, torch.float64)
    captured = forward(ids, torch.float32)

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
    out["greedy_ids"] = greedy_ids(ids)

    path = pathlib.Path(__file__).parent / "qwen3_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}")
    for name in ("block_0", "block_27", "norm", "logits"):
        print(f"  {name:10s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
