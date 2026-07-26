# pyright: basic
"""Capture LFM2.5-8B-A1B fp32 activations from transformers as ground truth.

The noise floor is the same graph replayed in fp64: at 24 layers the residual grows
along the trunk, so the floor is per tensor. fp32 of 8.3B is ~33GB and fp64 ~66GB, so
the two passes run as separate processes and only the captured tensors (5 tokens) are
carried between them.

Run, in order:
    uv run --with torch --with transformers --with safetensors --with accelerate \
        python packages/sideros/tests/fixtures/generate_lfm2_moe.py float32
    uv run ... generate_lfm2_moe.py float64
    uv run ... generate_lfm2_moe.py merge
After regenerating, update SHA256SUMS.lfm2_moe.
"""

import pathlib
import sys

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer, LFM2MoEForCausalLM
from transformers.models.lfm2_moe.modeling_lfm2_moe import apply_rotary_pos_emb

MODEL = "LiquidAI/LFM2.5-8B-A1B"
PROMPT = "The capital of France is"
ATTN_LAYER = 2  # first full_attention layer, also the first MoE layer
HERE = pathlib.Path(__file__).parent


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(ids: list[int], dtype: torch.dtype) -> dict[str, np.ndarray]:
    """Every tensor a parity test pins, in fp64 numpy whatever the graph's dtype."""
    # grouped_mm has no fp64 kernel; the eager expert loop works at any dtype.
    model = LFM2MoEForCausalLM.from_pretrained(MODEL, dtype=dtype, experts_implementation="eager")
    model.eval()
    captured: dict[str, np.ndarray] = {}

    conv_block = model.model.layers[0]
    attn_block = model.model.layers[ATTN_LAYER]
    handles = [
        model.model.embed_tokens.register_forward_hook(capture_into(captured, "embeddings")),
        model.model.embedding_norm.register_forward_hook(capture_into(captured, "norm")),
        # Layer 0: short conv + dense MLP.
        conv_block.operator_norm.register_forward_hook(capture_into(captured, "b0_ln_1")),
        conv_block.conv.register_forward_hook(capture_into(captured, "b0_conv")),
        conv_block.ffn_norm.register_forward_hook(capture_into(captured, "b0_ln_2")),
        conv_block.feed_forward.register_forward_hook(capture_into(captured, "b0_mlp")),
        # First attention layer, which is also the first MoE layer.
        attn_block.operator_norm.register_forward_hook(capture_into(captured, "b2_ln_1")),
        attn_block.self_attn.register_forward_hook(capture_into(captured, "b2_attn")),
        attn_block.self_attn.q_layernorm.register_forward_hook(capture_into(captured, "b2_q_norm")),
        attn_block.self_attn.k_layernorm.register_forward_hook(capture_into(captured, "b2_k_norm")),
        attn_block.ffn_norm.register_forward_hook(capture_into(captured, "b2_ln_2")),
        attn_block.feed_forward.register_forward_hook(capture_into(captured, "b2_moe")),
        attn_block.feed_forward.gate.register_forward_hook(capture_into(captured, "b2_router")),
    ]
    handles += [
        block.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, block in enumerate(model.model.layers)
    ]

    with torch.no_grad():
        captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()
    for handle in handles:
        handle.remove()

    # Replay HF's own rotary application on the captured normed projections: the same
    # ground truth, reachable from a boundary that is not a module.
    q = torch.tensor(captured["b2_q_norm"]).transpose(1, 2).to(dtype)
    k = torch.tensor(captured["b2_k_norm"]).transpose(1, 2).to(dtype)
    with torch.no_grad():
        cos, sin = model.model.pos_emb(q, position_ids=torch.arange(len(ids))[None])
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    captured["b2_q_rope"] = q_rope.double().numpy()
    captured["b2_k_rope"] = k_rope.double().numpy()

    if dtype is torch.float32:
        with torch.no_grad():
            greedy = model.generate(
                input_ids=torch.tensor([ids]), max_new_tokens=20, do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )
        captured["greedy_ids"] = greedy[0].numpy().astype(np.int32)
    return captured


def stage(dtype: torch.dtype, path: pathlib.Path) -> None:
    ids = AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"]
    captured = forward(ids, dtype)
    captured["input_ids"] = np.array(ids, dtype=np.int32)
    save_file({k: np.ascontiguousarray(v) for k, v in captured.items()}, path)
    print(f"{path}: {len(captured)} arrays")


def merge() -> None:
    exact = load_file(HERE / "_lfm2_moe_f64.safetensors")
    captured = load_file(HERE / "_lfm2_moe_f32.safetensors")
    out: dict[str, np.ndarray] = {}
    for name, tensor in captured.items():
        out[name] = tensor.astype(np.int32) if name.endswith("_ids") else tensor.astype(np.float32)
        if name in exact and not name.endswith("_ids"):
            out[f"noise.{name}"] = np.array(
                [relative_diff(tensor, exact[name])], dtype=np.float32
            )
    path = HERE / "lfm2_moe_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {out['input_ids'].tolist()}")
    print(f"  greedy -> {out['greedy_ids'].tolist()}")
    for name in ("block_0", f"block_{ATTN_LAYER}", "block_23", "norm", "logits"):
        print(f"  {name:10s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "merge":
        merge()
    else:
        stage(
            torch.float32 if what == "float32" else torch.float64,
            HERE / f"_lfm2_moe_{'f32' if what == 'float32' else 'f64'}.safetensors",
        )
