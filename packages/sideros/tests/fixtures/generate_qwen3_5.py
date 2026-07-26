# pyright: basic
"""Capture Qwen3.5-0.8B fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64, per tensor: at 24 layers the
residual grows along the trunk. Getting a true fp64 anchor means patching the modeling
file's internal float32 casts (RMSNorm, RMSNormGated, the chunked delta rule).

Run: uv run --with torch --with transformers --with safetensors \
     python packages/sideros/tests/fixtures/generate_qwen3_5.py
After regenerating, update SHA256SUMS.qwen3_5.
"""

import gc
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
from safetensors.numpy import save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT = "The capital of France is"
LINEAR_LAYER = 0
ATTN_LAYER = 3

ORIGINAL_CHUNK_RULE = modeling.torch_chunk_gated_delta_rule


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def dtype_preserving_rmsnorm(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + self.eps) * (1.0 + self.weight)


def dtype_preserving_rmsnorm_gated(self, hidden_states, gate=None):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states * F.silu(gate)


def dtype_preserving_chunk_rule(query, key, value, g, beta, **kwargs):
    """The rule casts everything to float32 internally; keep float64 float64."""
    original = torch.Tensor.to

    def to(self, *args, **kw):
        if args and args[0] is torch.float32 and self.dtype is torch.float64:
            return self
        return original(self, *args, **kw)

    torch.Tensor.to = to
    try:
        return ORIGINAL_CHUNK_RULE(query, key, value, g, beta, **kwargs)
    finally:
        torch.Tensor.to = original


def forward(ids: list[int], dtype: torch.dtype) -> dict[str, np.ndarray]:
    """Every tensor a parity test pins, in fp64 numpy regardless of the graph dtype."""
    exact = dtype is torch.float64
    saved = (
        modeling.Qwen3_5RMSNorm.forward,
        modeling.Qwen3_5RMSNormGated.forward,
        modeling.torch_chunk_gated_delta_rule,
    )
    if exact:
        modeling.Qwen3_5RMSNorm.forward = dtype_preserving_rmsnorm
        modeling.Qwen3_5RMSNormGated.forward = dtype_preserving_rmsnorm_gated
        modeling.torch_chunk_gated_delta_rule = dtype_preserving_chunk_rule

    captured: dict[str, np.ndarray] = {}
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=dtype)
        model.eval()
        text = model.model.language_model
        linear = text.layers[LINEAR_LAYER]
        attention = text.layers[ATTN_LAYER]

        handles = [
            text.embed_tokens.register_forward_hook(capture_into(captured, "embeddings")),
            text.norm.register_forward_hook(capture_into(captured, "norm")),
            linear.input_layernorm.register_forward_hook(capture_into(captured, "b0_ln_1")),
            linear.linear_attn.register_forward_hook(capture_into(captured, "b0_mixer")),
            linear.post_attention_layernorm.register_forward_hook(
                capture_into(captured, "b0_ln_2")
            ),
            linear.mlp.register_forward_hook(capture_into(captured, "b0_mlp")),
            attention.input_layernorm.register_forward_hook(capture_into(captured, "b3_ln_1")),
            attention.self_attn.q_norm.register_forward_hook(capture_into(captured, "b3_q_norm")),
            attention.self_attn.k_norm.register_forward_hook(capture_into(captured, "b3_k_norm")),
            attention.self_attn.register_forward_hook(capture_into(captured, "b3_attn")),
            attention.post_attention_layernorm.register_forward_hook(
                capture_into(captured, "b3_ln_2")
            ),
            attention.mlp.register_forward_hook(capture_into(captured, "b3_mlp")),
        ]
        handles += [
            block.register_forward_hook(capture_into(captured, f"block_{i}"))
            for i, block in enumerate(text.layers)
        ]

        # The delta rule is a plain instance attribute, not a module: wrap it to record
        # its inputs (post conv+silu, pre l2norm) and its outputs (out, final state).
        original_rule = linear.linear_attn.chunk_gated_delta_rule

        def recording_rule(query, key, value, g, beta, **kwargs):
            for name, tensor in (("q", query), ("k", key), ("v", value), ("g", g),
                                 ("beta", beta)):
                captured[f"b0_rule_{name}"] = tensor.detach().double().numpy()
            out, state = original_rule(query, key, value, g=g, beta=beta, **kwargs)
            captured["b0_rule_out"] = out.detach().double().numpy()
            captured["b0_rule_state"] = state.detach().double().numpy()
            return out, state

        linear.linear_attn.chunk_gated_delta_rule = recording_rule
        with torch.no_grad():
            captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()
        linear.linear_attn.chunk_gated_delta_rule = original_rule

        # Replay HF's rotary application on the captured normed projections: same ground
        # truth, reachable from a boundary that is not a module.
        q = torch.tensor(captured["b3_q_norm"]).transpose(1, 2).to(dtype)
        k = torch.tensor(captured["b3_k_norm"]).transpose(1, 2).to(dtype)
        with torch.no_grad():
            cos, sin = text.rotary_emb(q, torch.arange(len(ids))[None])
            q_rope, k_rope = modeling.apply_rotary_pos_emb(q, k, cos, sin)
        captured["b3_q_rope"] = q_rope.double().numpy()
        captured["b3_k_rope"] = k_rope.double().numpy()

        for handle in handles:
            handle.remove()
        del model
        gc.collect()
    finally:
        (
            modeling.Qwen3_5RMSNorm.forward,
            modeling.Qwen3_5RMSNormGated.forward,
            modeling.torch_chunk_gated_delta_rule,
        ) = saved
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]), max_new_tokens=20, do_sample=False,
            pad_token_id=model.config.text_config.eos_token_id,
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

    path = pathlib.Path(__file__).parent / "qwen3_5_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}")
    print(f"  greedy -> {generated.tolist()}")
    for name in ("block_0", "block_23", "b0_rule_out", "norm", "logits"):
        print(f"  {name:12s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()
