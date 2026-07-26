"""rope_epilogue against the ops path it replaces, at Qwen3-30B-A3B shapes.

The replica is the complete tail of `Qwen3Attention.__call__`: the real fused qkv
projection of a normed hidden state (so the dots land at checkpoint magnitudes),
then split/reshape, `q_norm`/`k_norm`, `mx.fast.rope`. In bf16 the kernel is
bit-exact with that chain; the fp32 template of the same kernel is what the 1e-5
comparison exercises.
"""

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from conftest import relative_diff

from sideros.core.kernels.rope_epilogue import _KERNEL, _SOURCE, rope_epilogue
from sideros.core.mxcompat import metal_kernel
from sideros.models.qwen3_moe import Qwen3Attention, Qwen3MoEConfig

if TYPE_CHECKING:
    from sideros.core.mxcompat import MetalKernel

CONFIG = Qwen3MoEConfig(
    hidden_size=2048,
    num_hidden_layers=48,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    vocab_size=151936,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    tie_word_embeddings=False,
    moe_intermediate_size=768,
    num_experts=128,
    num_experts_per_tok=8,
    norm_topk_prob=True,
    quantization=(64, 4),
)


def _replica(dtype: mx.Dtype, seed: int) -> tuple[Qwen3Attention, mx.array]:
    """Attention with checkpoint-scale weights and the fused qkv of one normed token."""
    rng = np.random.default_rng(seed)
    attention = Qwen3Attention(CONFIG)
    hidden, head_dim = CONFIG.hidden_size, CONFIG.head_dim
    width = (CONFIG.num_attention_heads + 2 * CONFIG.num_key_value_heads) * head_dim
    scale = 1.0 / math.sqrt(hidden)
    attention.qkv_proj.weight = mx.array(
        (rng.standard_normal((width, hidden)) * scale).astype(np.float32)
    ).astype(dtype)
    attention.q_norm.weight = mx.array(
        (1.0 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32)
    ).astype(dtype)
    attention.k_norm.weight = mx.array(
        (1.0 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32)
    ).astype(dtype)
    x = mx.array(rng.standard_normal((1, 1, hidden)).astype(np.float32)).astype(dtype)
    return attention, attention.qkv_proj(x)


def _ops_path(attention: Qwen3Attention, fused: mx.array, offset: int) -> tuple[mx.array, mx.array]:
    head_dim = CONFIG.head_dim
    query_width = CONFIG.num_attention_heads * head_dim
    kv_width = CONFIG.num_key_value_heads * head_dim
    q, k, _ = (
        part.reshape(1, 1, count, head_dim).transpose(0, 2, 1, 3)
        for part, count in zip(
            mx.split(fused, [query_width, query_width + kv_width], axis=-1),
            (CONFIG.num_attention_heads, CONFIG.num_key_value_heads, CONFIG.num_key_value_heads),
            strict=True,
        )
    )
    rope = attention._rope
    return (
        rope(attention.q_norm(q), offset).reshape(-1),
        rope(attention.k_norm(k), offset).reshape(-1),
    )


def _kernel_path(
    kernel: "MetalKernel", attention: Qwen3Attention, fused: mx.array, offset: int
) -> tuple[mx.array, mx.array]:
    out = kernel(
        inputs=[
            fused.reshape(-1),
            attention.q_norm.weight,
            attention.k_norm.weight,
            mx.array([offset], dtype=mx.int32),
            mx.array([math.log2(CONFIG.rope_theta)], dtype=mx.float32),
            mx.array([CONFIG.rms_norm_eps], dtype=mx.float32),
        ],
        template=[
            ("T", fused.dtype),
            ("QHEADS", CONFIG.num_attention_heads),
            ("HDIM", CONFIG.head_dim),
        ],
        grid=((CONFIG.num_attention_heads + CONFIG.num_key_value_heads) * CONFIG.head_dim, 1, 1),
        threadgroup=(CONFIG.head_dim, 1, 1),
        output_shapes=[
            (CONFIG.num_attention_heads * CONFIG.head_dim,),
            (CONFIG.num_key_value_heads * CONFIG.head_dim,),
        ],
        output_dtypes=[fused.dtype, fused.dtype],
    )
    return out[0], out[1]


OFFSETS = (0, 1, 137, 4095)


def test_matches_op_chain_fp32() -> None:
    attention, fused = _replica(mx.float32, seed=11)
    for offset in OFFSETS:
        q, k = rope_epilogue(
            fused,
            query_heads=CONFIG.num_attention_heads,
            kv_heads=CONFIG.num_key_value_heads,
            head_dim=CONFIG.head_dim,
            q_norm=attention.q_norm.weight,
            k_norm=attention.k_norm.weight,
            offset=offset,
            base=CONFIG.rope_theta,
            eps=CONFIG.rms_norm_eps,
        )
        ref_q, ref_k = _ops_path(attention, fused, offset)
        assert relative_diff(q, ref_q) < 1e-5
        assert relative_diff(k, ref_k) < 1e-5


def test_bit_exact_bf16() -> None:
    """No floor to measure: the norm rounds to T before the weight multiply and the
    angle is mlx's exp2/fast::cos, so every bf16 bit is the chain's."""
    attention, fused = _replica(mx.bfloat16, seed=5)
    for offset in OFFSETS:
        q, k = rope_epilogue(
            fused,
            query_heads=CONFIG.num_attention_heads,
            kv_heads=CONFIG.num_key_value_heads,
            head_dim=CONFIG.head_dim,
            q_norm=attention.q_norm.weight,
            k_norm=attention.k_norm.weight,
            offset=offset,
            base=CONFIG.rope_theta,
            eps=CONFIG.rms_norm_eps,
        )
        ref_q, ref_k = _ops_path(attention, fused, offset)
        assert bool(mx.array_equal(q, ref_q))
        assert bool(mx.array_equal(k, ref_k))


MUTATIONS = {
    "rope_sign": (
        "(other * sn + vn * c) : (vn * c - other * sn)",
        "(other * sn + vn * c) : (vn * c + other * sn)",
    ),
    "norm_weights_swapped": ("isq ? QNW : KNW", "isq ? KNW : QNW"),
    "offset_ignored": ("(float)OFF[0] * metal::exp2", "1.0f * metal::exp2"),
    # eps is not on the menu: at 1e-6 against a mean square of ~1 it moves the norm by
    # 5e-7 relative, under both the bf16 ulp and the fp32 tolerance.
    "rotate_halves_swapped": (
        "hi ? tid - HDIM / 2 : tid + HDIM / 2",
        "hi ? tid + HDIM / 2 : tid - HDIM / 2",
    ),
    "norm_rounding_removed": (
        "(float)(T)((float)nw[tid] * (float)(T)(v * hr))",
        "(float)nw[tid] * v * hr",
    ),
}


def test_mutations_are_caught() -> None:
    attention, fused = _replica(mx.bfloat16, seed=5)
    offset = 137
    ref_q, ref_k = _ops_path(attention, fused, offset)
    intact = _kernel_path(_KERNEL, attention, fused, offset)
    assert bool(mx.array_equal(intact[0], ref_q))
    for name, (old, new) in MUTATIONS.items():
        assert old in _SOURCE, name
        broken = metal_kernel(
            name=f"rope_epilogue_{name}",
            input_names=["F", "QNW", "KNW", "OFF", "LOGBASE", "EPS"],
            output_names=["Q", "K"],
            source=_SOURCE.replace(old, new),
        )
        q, k = _kernel_path(broken, attention, fused, offset)
        matches = mx.array_equal(q, ref_q) and mx.array_equal(k, ref_k)
        assert not bool(matches), name
