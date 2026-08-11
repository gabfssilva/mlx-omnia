"""The one-token attention step of a model with learned sinks, split-K flash-decode.

A lone query attends to the whole cache, so this is a flash-decode — but a single
threadgroup per query head leaves a 40-core GPU idle (64 threadgroups, one simdgroup
each, grinding a serial ~600-key loop): latency, not bandwidth. So the keys are split
`_SPLITS` ways: each `(head, split)` threadgroup runs the online softmax over its slice
and writes a partial `(max, denominator, accumulator)`, and a cheap combine pass merges
the slices under one global max and folds in the sink. The split changes only how the
float32 sums are grouped.

q, k and v are read in the model's dtype; the score, the softmax and the accumulation
stay in float32 and only the context rounds back to `T`. A masked key (additive `-inf`,
the sliding band's older half) is skipped rather than added; a slice that is wholly
masked contributes nothing (`max == -inf` rescales to zero rather than
`exp(-inf - -inf)` NaN).

Ported from `.legacy/Sources/Sideros/Core/SinkAttention.swift`. Three adaptations:

- the Swift rounded the score to `T` after the dot, matching the Swift-era hand-written
  attention whose q·kᵀ was a bf16 matmul. The Python model's reference is
  `mx.fast.scaled_dot_product_attention(sinks=…)`, which keeps the score in float32:
  measured, the rounded score sits at ~4x the fixture floor from that reference while
  the float32 score lands within 5.6e-5 of it (and exactly on the floor against the
  float32 golden). The rounding is dropped, so the kernel replaces the reference in
  place instead of tracking a rounding the model no longer has;
- the key/value reads index the buffers directly instead of taking a `device` pointer:
  mlx picks the address space of an input by its size, so an explicit qualifier only
  compiles for some shapes;
- the per-call constants that never change (`GROUPS`, `SPLITS`) are template parameters
  rather than scalar inputs.

`keys`/`values` arrive as `KVCache.update_and_fetch` returns them: a slice of the 256-row
block buffer, so the head stride is the buffer's *capacity*, not `key_length`. Reshaping
them to 3-D would make mlx materialize `kv_heads·key_length·64` per step — a copy the
`mx.fast.scaled_dot_product_attention` baseline does not pay. Instead the kernel is built
with `ensure_row_contiguous=False` and reads the head/row strides mlx passes as
`K_strides`/`V_strides` (mlx supplies `{name}_shape`/`_strides`/`_ndim` for any input
whose name appears that way in the source). The last axis is still assumed packed —
head_dim is the contiguous dimension in every layout this kernel accepts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from sideros.core.cache import KVCache
from sideros.core.kernels.attention.default import (
    DefaultAttentionStep,
    qk_norm,
    rotate,
    split_heads,
    write_position,
)
from sideros.core.kernels.attention.kernel import Angles, AttentionCache
from sideros.core.mxcompat import metal_kernel

_SPLITS = 8
_HEAD_DIM = 64

_PARTIAL_SOURCE = """
    uint h = threadgroup_position_in_grid.y;
    uint si = threadgroup_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint kv = h / (uint)GROUPS;
    uint kl = (uint)KEYLEN;
    uint S = (uint)SPLITS;
    constexpr uint hd = 64;
    uint chunk = (kl + S - 1) / S;
    uint j0 = si * chunk;
    uint j1 = metal::min(j0 + chunk, kl);
    size_t khead = (size_t)K_strides[1], krow = (size_t)K_strides[2];
    size_t vhead = (size_t)V_strides[1], vrow = (size_t)V_strides[2];

    threadgroup float qs[hd];
    if (lane < hd / 2) {
        qs[lane] = (float)Q[h * hd + lane];
        qs[lane + 32] = (float)Q[h * hd + lane + 32];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f;
    float acc[hd];
    for (uint d = 0; d < hd; d++) acc[d] = 0.0f;

    for (uint j = j0 + lane; j < j1; j += 32) {
        float mk = MASK[j];
        if (metal::isinf(mk)) continue;
        size_t kp = kv * khead + (size_t)j * krow;
        size_t vp = kv * vhead + (size_t)j * vrow;
        float dot = 0.0f;
        for (uint d = 0; d < hd; d++) dot += qs[d] * (float)K[kp + d];
        float score = dot * SCALE + mk;
        float mn = metal::max(m, score);
        float corr = metal::exp(m - mn);
        float p = metal::exp(score - mn);
        l = l * corr + p;
        for (uint d = 0; d < hd; d++) acc[d] = acc[d] * corr + p * (float)V[vp + d];
        m = mn;
    }

    float mg = simd_max(m);
    float corr = (mg == -INFINITY) ? 0.0f : metal::exp(m - mg);
    l = simd_sum(l * corr);
    for (uint d = 0; d < hd; d++) acc[d] = simd_sum(acc[d] * corr);
    if (lane == 0) {
        ML[(h * S + si) * 2 + 0] = mg;
        ML[(h * S + si) * 2 + 1] = l;
    }
    for (uint d = lane; d < hd; d += 32) AP[(size_t)(h * S + si) * hd + d] = acc[d];
"""

_COMBINE_SOURCE = """
    uint h = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint S = (uint)SPLITS;
    constexpr uint hd = 64;

    float gm = -INFINITY;
    for (uint si = 0; si < S; si++) gm = metal::max(gm, ML[(h * S + si) * 2 + 0]);
    gm = metal::max(gm, (float)SINKS[h]);

    float L = metal::exp((float)SINKS[h] - gm);
    for (uint si = 0; si < S; si++) {
        float mi = ML[(h * S + si) * 2 + 0];
        L += ML[(h * S + si) * 2 + 1] * metal::exp(mi - gm);
    }
    for (uint d = lane; d < hd; d += 32) {
        float a = 0.0f;
        for (uint si = 0; si < S; si++)
            a += AP[(size_t)(h * S + si) * hd + d] * metal::exp(ML[(h * S + si) * 2 + 0] - gm);
        OUT[h * hd + d] = (T)(a / L);
    }
"""

SinkAttention = Callable[
    [mx.array, mx.array, mx.array, mx.array, mx.array | None, float], mx.array
]


def _build(name: str, partial_source: str, combine_source: str) -> SinkAttention:
    """The dispatch pair plus its host side. Parameterized by source so a mutation test
    can rebuild a broken variant without duplicating the host code."""
    partial = metal_kernel(
        name=f"{name}_partial",
        input_names=["Q", "K", "V", "MASK", "KEYLEN", "SCALE"],
        output_names=["ML", "AP"],
        source=partial_source,
        ensure_row_contiguous=False,
    )
    combine = metal_kernel(
        name=f"{name}_combine",
        input_names=["ML", "AP", "SINKS"],
        output_names=["OUT"],
        source=combine_source,
    )

    def run(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        sinks: mx.array,
        mask: mx.array | None,
        scale: float,
    ) -> mx.array:
        heads, kv_heads, key_length = queries.shape[1], keys.shape[1], keys.shape[2]
        dtype = queries.dtype
        additive = _additive_mask(mask, key_length)
        partials = partial(
            inputs=[
                queries.reshape(heads, _HEAD_DIM),
                keys,
                values,
                additive,
                mx.array(key_length, dtype=mx.int32),
                mx.array(scale, dtype=mx.float32),
            ],
            template=[("T", dtype), ("GROUPS", heads // kv_heads), ("SPLITS", _SPLITS)],
            grid=(32, heads, _SPLITS),
            threadgroup=(32, 1, 1),
            output_shapes=[(heads * _SPLITS, 2), (heads * _SPLITS, _HEAD_DIM)],
            output_dtypes=[mx.float32, mx.float32],
        )
        out = combine(
            inputs=[partials[0], partials[1], sinks],
            template=[("T", dtype), ("SPLITS", _SPLITS)],
            grid=(32, heads, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(heads, _HEAD_DIM)],
            output_dtypes=[dtype],
        )
        return out[0].reshape(1, heads, 1, _HEAD_DIM)

    return run


def _additive_mask(mask: mx.array | None, key_length: int) -> mx.array:
    """The model's boolean band becomes the kernel's additive `-inf`; a full layer
    passes `None` (a lone query attends to everything)."""
    if mask is None:
        return mx.zeros((key_length,), dtype=mx.float32)
    flat = mask.reshape(-1)
    assert flat.size == key_length
    if flat.dtype == mx.bool_:
        return mx.where(flat, mx.array(0.0), mx.array(-mx.inf)).astype(mx.float32)
    return flat.astype(mx.float32)


_KERNEL = _build("sink_attention", _PARTIAL_SOURCE, _COMBINE_SOURCE)


def sink_attention_applies(queries: mx.array, keys: mx.array) -> bool:
    """One query row per head, head_dim 64, GQA by whole groups — the shape the kernel
    is written for. Anything else stays on `mx.fast.scaled_dot_product_attention`."""
    return (
        queries.ndim == 4
        and queries.shape[0] == 1
        and queries.shape[2] == 1
        and queries.shape[3] == _HEAD_DIM
        and keys.shape[3] == _HEAD_DIM
        and queries.shape[1] % keys.shape[1] == 0
    )


def sink_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    sinks: mx.array,
    mask: mx.array | None,
    scale: float,
) -> mx.array:
    """A single query per head against the whole cache, the sink in the denominator.

    `queries` [1, heads, 1, 64], `keys`/`values` [1, kv_heads, key_length, 64], `sinks`
    [heads], `mask` the model's boolean band over the keys (or `None` for a full layer).
    Returns the context [1, heads, 1, 64], the shape
    `mx.fast.scaled_dot_product_attention` returns.
    """
    assert sink_attention_applies(queries, keys)
    assert keys.dtype == queries.dtype and values.dtype == queries.dtype
    assert sinks.dtype == queries.dtype
    return _KERNEL(queries, keys, values, sinks, mask, scale)


@dataclass(frozen=True)
class SinkAttentionStep:
    """The decode step of a layer with learned sinks: norm and rotation in ops, then the
    cache write and the split-K flash-decode on `sink_attention`.

    The fused boundary stops before the rotation here — the kernel takes q/k already
    roped, the way `mx.fast.scaled_dot_product_attention` does. A string mask becomes
    `None`: a single query row sees every key a causal mask would allow. Prefill
    (`T != 1`) runs on the ops chain.
    """

    cache: KVCache
    heads: int
    kv_heads: int
    head_dim: int
    scale: float
    query_weight: mx.array | None
    key_weight: mx.array | None
    eps: float
    rotary_pairs: int
    mscale: float
    freqs: mx.array | None
    base: float
    sinks: mx.array
    fallback: DefaultAttentionStep

    @classmethod
    def build(
        cls,
        cache: AttentionCache,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        scale: float,
        dtype: mx.Dtype,
        query_weight: mx.array | None,
        key_weight: mx.array | None,
        eps: float,
        rotary_pairs: int,
        mscale: float,
        angles: Angles | None,
        freqs: mx.array | None,
        base: float,
        sinks: mx.array | None,
    ) -> Self | None:
        if not isinstance(cache, KVCache) or sinks is None:
            return None
        if sinks.dtype != dtype or sinks.shape != (heads,):
            return None
        if head_dim != _HEAD_DIM or heads % kv_heads != 0:
            return None
        return cls(
            cache,
            heads,
            kv_heads,
            head_dim,
            scale,
            query_weight,
            key_weight,
            eps,
            rotary_pairs,
            mscale,
            freqs,
            base,
            sinks,
            DefaultAttentionStep.build(
                cache,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                scale=scale,
                dtype=dtype,
                query_weight=query_weight,
                key_weight=key_weight,
                eps=eps,
                rotary_pairs=rotary_pairs,
                mscale=mscale,
                angles=angles,
                freqs=freqs,
                base=base,
                sinks=sinks,
            ),
        )

    def __call__(
        self,
        raw_queries: mx.array,
        raw_keys: mx.array,
        raw_values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        if raw_queries.shape[1] != 1:
            return self.fallback(raw_queries, raw_keys, raw_values, mask)
        offset = write_position(self.cache)
        queries = split_heads(raw_queries, self.heads, self.head_dim)
        keys = split_heads(raw_keys, self.kv_heads, self.head_dim)
        values = split_heads(raw_values, self.kv_heads, self.head_dim)
        queries = self._rotate(qk_norm(queries, self.query_weight, self.eps), offset)
        keys = self._rotate(qk_norm(keys, self.key_weight, self.eps), offset)
        cached_keys, cached_values = self.cache.update_and_fetch(keys, values)
        band = None if isinstance(mask, str) else mask
        return sink_attention(
            queries, cached_keys, cached_values, self.sinks, band, self.scale
        )

    def _rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return rotate(
            x,
            rotary_pairs=self.rotary_pairs,
            freqs=self.freqs,
            base=self.base,
            mscale=self.mscale,
            offset=offset,
        )
