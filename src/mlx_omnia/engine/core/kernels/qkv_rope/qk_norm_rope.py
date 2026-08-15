"""Per-head q/k RMSNorm + a table-driven rotary, one dispatch for every head of a
projection — decode (one token) and prefill (a whole segment), with and without a
pre-scale on the rotating half.

The op chain these replace is six dispatches on the partial-rotary path (`rms_norm` x2,
the copy each partial `fast.rope` materializes first x2, the rope itself x2) and four on
the full-rotary one. One threadgroup is one simdgroup and owns one head: lane `l` holds
the contiguous block `[l * per_lane, (l + 1) * per_lane)` of the head, so the sum of
squares is a `simd_sum` and the rotary partner is a `simd_shuffle` — no threadgroup
memory, no barrier.

The rotation is half-split *inside the rotary section*: element `p` couples with
`p + rotary_pairs`, for `p < rotary_pairs`. With `2 * rotary_pairs < head_dim` the
remaining `head_dim - 2 * rotary_pairs` elements do not rotate at all and are written
verbatim from the norm — the partial-rotary boundary is `2 * rotary_pairs`, and it is a
*lane* boundary: lanes `[0, rotary_lanes)` write both halves of every pair they own,
lanes `[rotary_lanes, 2 * rotary_lanes)` write nothing (their elements were the second
half of a pair, already written by their shuffle partner), and lanes
`[2 * rotary_lanes, 32)` copy the tail through. `rotary_lanes = rotary_pairs / per_lane`.

`angles` is precomputed, never re-derived: a row of `2 * rotary_pairs` floats, cosines
then sines. Decode reads the single row; prefill indexes row `offset + t` of a table.
Reusing the same floats the op chain would have used is what keeps the kernel exact
rather than close — a host-side re-derivation of the frequencies is a different number.
Hand the decode entry point the row itself, never `table[position : position + 1]`: a
row slice is row-contiguous *by flags* with a non-zero data offset, so neither
`ensure_row_contiguous` nor `mx.contiguous` copies it. Prefill has no such hazard — it
takes the whole table and the row index as `offsets`.

The scaled variants carry an `mscale` (a YaRN-style attention factor) applied to the
rotary *inputs* rather than to the angles, and applied through a round-trip in T:
`float(T(x * T(mscale)))`. That is the rounding the op it replaces performs, and it is
load-bearing. The tail is never scaled.

Two roundings are copied from `rms_norm` and are equally load-bearing:
`w[i] * T(x[i] * inv_rms)` rounds the normalized value to T *before* the weight multiply,
and the accumulation is `per_lane` squares in index order then a `simd_sum`, which is
`rms_single_row` with `N_READS = per_lane` over a 32-thread group.

Relation to `rope_epilogue`: that kernel is the same fusion for a *fused qkv* projection
of one token — it slices q and k out of a single tensor, derives the angle in-kernel from
`offset` and `log2(base)` with `metal::fast::cos/sin`, rotates the *whole* head with the
naive half-split, and has no pre-scale. These take q and k as separate tensors, take the
angles as an input, and add the partial-rotary tail and the mscale round-trip. Neither
subsumes the other; a model on the full-rotary decode path with a fused qkv still wants
`rope_epilogue`.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.qkv_rope.default import DefaultQkvRope
from mlx_omnia.engine.core.kernels.qkv_rope.kernel import (
    Angles,
    Leaf,
    Projection,
    QkvRopeStrategy,
    Rotation,
)
from mlx_omnia.engine.core.mxcompat import metal_kernel

_DECODE_SOURCE = """
    constexpr uint head_dim = HEAD_DIM;
    constexpr uint rotary_dims = 2 * ROTARY_PAIRS;
    constexpr uint rotary_pairs = ROTARY_PAIRS;
    constexpr uint query_heads = QUERY_HEADS;
    constexpr uint per_lane = head_dim / 32;
    constexpr uint rotary_lanes = rotary_pairs / per_lane;

    uint head = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;


    const device T* input;
    const device T* weight;
    if (head < query_heads) {
        input = raw_queries + head * head_dim;
        weight = query_weight;
    } else {
        input = raw_keys + (head - query_heads) * head_dim;
        weight = key_weight;
    }

    uint base = lane * per_lane;
    thread T normalized[per_lane];
    float sum = 0.0f;
    for (uint i = 0; i < per_lane; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / float(head_dim) + eps[0]);

    for (uint i = 0; i < per_lane; ++i) {
        normalized[i] =
            weight[base + i] *
            T(float(input[base + i]) * inverse_rms);
    }

    thread float paired[per_lane];
    for (uint i = 0; i < per_lane; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ rotary_lanes);
    }

    device T* output =
        head < query_heads
        ? queries + head * head_dim
        : keys + (head - query_heads) * head_dim;
    if (lane < rotary_lanes) {
        T rounded_mscale = T(mscale[0]);
        for (uint i = 0; i < per_lane; ++i) {
            uint pair = base + i;
            float first =
                float(T(normalized[i] * rounded_mscale));
            float second =
                float(T(T(paired[i]) * rounded_mscale));
            float cosine = angles[pair];
            float sine = angles[pair + rotary_pairs];
            output[pair] = T(first * cosine - second * sine);
            output[pair + rotary_pairs] =
                T(first * sine + second * cosine);
        }
    } else if (lane >= 2 * rotary_lanes) {
        for (uint i = 0; i < per_lane; ++i) {
            output[base + i] = normalized[i];
        }
    }
"""

_DECODE_INPUTS = [
    "raw_queries",
    "raw_keys",
    "query_weight",
    "key_weight",
    "angles",
    "eps",
    "mscale",
]
_OUTPUTS = ["queries", "keys"]

_DECODE_KERNEL = metal_kernel(
    name="qk_norm_rope_decode",
    input_names=_DECODE_INPUTS,
    output_names=_OUTPUTS,
    source=_DECODE_SOURCE,
)

_PREFILL_SCALED_SOURCE = """
    constexpr uint head_dim = HEAD_DIM;
    constexpr uint rotary_pairs = ROTARY_PAIRS;
    constexpr uint query_heads = QUERY_HEADS;
    constexpr uint kv_heads = KV_HEADS;
    constexpr uint per_lane = head_dim / 32;
    constexpr uint rotary_lanes = rotary_pairs / per_lane;

    uint t = threadgroup_position_in_grid.y;
    uint length = threadgroups_per_grid.y;
    uint head = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;

    const device T* input;
    const device T* weight;
    device T* output;
    if (head < query_heads) {
        input = raw_queries + (t * query_heads + head) * head_dim;
        weight = query_weight;
        output = queries + (head * length + t) * head_dim;
    } else {
        uint khead = head - query_heads;
        input = raw_keys + (t * kv_heads + khead) * head_dim;
        weight = key_weight;
        output = keys + (khead * length + t) * head_dim;
    }

    uint base = lane * per_lane;
    thread T normalized[per_lane];
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < per_lane; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / float(head_dim) + eps[0]);

    #pragma clang loop unroll(full)
    for (uint i = 0; i < per_lane; ++i) {
        normalized[i] =
            weight[base + i] *
            T(float(input[base + i]) * inverse_rms);
    }

    thread float paired[per_lane];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < per_lane; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ rotary_lanes);
    }

    const device float* angle_row =
        angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
    if (lane < rotary_lanes) {
        T rounded_mscale = T(mscale[0]);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < per_lane; ++i) {
            uint pair = base + i;
            float first =
                float(T(normalized[i] * rounded_mscale));
            float second =
                float(T(T(paired[i]) * rounded_mscale));
            float cosine = angle_row[pair];
            float sine = angle_row[pair + rotary_pairs];
            output[pair] = T(first * cosine - second * sine);
            output[pair + rotary_pairs] =
                T(first * sine + second * cosine);
        }
    } else if (lane >= 2 * rotary_lanes) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < per_lane; ++i) {
            output[base + i] = normalized[i];
        }
    }
"""

_PREFILL_SCALED_INPUTS = [
    "raw_queries",
    "raw_keys",
    "query_weight",
    "key_weight",
    "angles",
    "offsets",
    "eps",
    "mscale",
]

_PREFILL_SCALED_KERNEL = metal_kernel(
    name="qk_norm_rope_prefill_scaled",
    input_names=_PREFILL_SCALED_INPUTS,
    output_names=_OUTPUTS,
    source=_PREFILL_SCALED_SOURCE,
)

_PREFILL_SOURCE = """
    constexpr uint head_dim = HEAD_DIM;
    constexpr uint rotary_pairs = ROTARY_PAIRS;
    constexpr uint query_heads = QUERY_HEADS;
    constexpr uint kv_heads = KV_HEADS;
    constexpr uint per_lane = head_dim / 32;
    constexpr uint rotary_lanes = rotary_pairs / per_lane;

    uint t = threadgroup_position_in_grid.y;
    uint length = threadgroups_per_grid.y;
    uint head = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;

    const device T* input;
    const device T* weight;
    device T* output;
    if (head < query_heads) {
        input = raw_queries + (t * query_heads + head) * head_dim;
        weight = query_weight;
        output = queries + (head * length + t) * head_dim;
    } else {
        uint khead = head - query_heads;
        input = raw_keys + (t * kv_heads + khead) * head_dim;
        weight = key_weight;
        output = keys + (khead * length + t) * head_dim;
    }

    uint base = lane * per_lane;
    thread T normalized[per_lane];
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < per_lane; ++i) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = metal::precise::rsqrt(sum / float(head_dim) + eps[0]);

    #pragma clang loop unroll(full)
    for (uint i = 0; i < per_lane; ++i) {
        normalized[i] =
            weight[base + i] *
            T(float(input[base + i]) * inverse_rms);
    }

    thread float paired[per_lane];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < per_lane; ++i) {
        paired[i] = simd_shuffle(float(normalized[i]), lane ^ rotary_lanes);
    }

    const device float* angle_row =
        angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);
    if (lane < rotary_lanes) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < per_lane; ++i) {
            uint pair = base + i;
            float first = float(normalized[i]);
            float second = paired[i];
            float cosine = angle_row[pair];
            float sine = angle_row[pair + rotary_pairs];
            output[pair] = T(first * cosine - second * sine);
            output[pair + rotary_pairs] =
                T(first * sine + second * cosine);
        }
    }
"""

_PREFILL_INPUTS = [
    "raw_queries",
    "raw_keys",
    "query_weight",
    "key_weight",
    "angles",
    "offsets",
    "eps",
]

_PREFILL_KERNEL = metal_kernel(
    name="qk_norm_rope_prefill",
    input_names=_PREFILL_INPUTS,
    output_names=_OUTPUTS,
    source=_PREFILL_SOURCE,
)


def _lane_layout(head_dim: int, rotary_pairs: int) -> bool:
    """The head must tile over one simdgroup as `per_lane` contiguous elements, and the
    rotary partner of element `p` (which is `p + rotary_pairs`) must sit on lane
    `lane ^ rotary_lanes` — so `rotary_pairs` is a whole number of lanes and that number
    is a power of two. `2 * rotary_pairs <= head_dim` is the rotary section fitting in
    the head."""
    if head_dim % 32 != 0 or head_dim < 32:
        return False
    per_lane = head_dim // 32
    if rotary_pairs <= 0 or rotary_pairs % per_lane != 0:
        return False
    rotary_lanes = rotary_pairs // per_lane
    return rotary_lanes & (rotary_lanes - 1) == 0 and 2 * rotary_pairs <= head_dim


def _check(
    raw_queries: mx.array,
    raw_keys: mx.array,
    query_weight: mx.array,
    key_weight: mx.array,
    angles: mx.array,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_pairs: int,
) -> int:
    """Shared preconditions; returns the segment length read off `raw_queries`."""
    dtype = raw_queries.dtype
    assert raw_keys.dtype == dtype
    assert query_weight.dtype == dtype and key_weight.dtype == dtype
    assert query_weight.size == head_dim and key_weight.size == head_dim
    assert angles.dtype == mx.float32
    length, remainder = divmod(raw_queries.size, query_heads * head_dim)
    assert remainder == 0 and length > 0
    assert raw_keys.size == length * kv_heads * head_dim
    assert 2 * rotary_pairs <= head_dim
    return length


def qk_norm_rope_decode_applies(head_dim: int, rotary_pairs: int) -> bool:
    """One simdgroup per head, `head_dim / 32` contiguous elements per lane; the rotary
    section may be shorter than the head, and the tail is copied through."""
    return _lane_layout(head_dim, rotary_pairs)


def qk_norm_rope_decode(
    raw_queries: mx.array,
    raw_keys: mx.array,
    *,
    query_weight: mx.array,
    key_weight: mx.array,
    angles: mx.array,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_pairs: int,
    eps: float,
    mscale: float,
) -> tuple[mx.array, mx.array]:
    """One token: `raw_queries` [query_heads * head_dim] and `raw_keys`
    [kv_heads * head_dim], head-major, normed per head and rotated by the single
    `angles` row of `2 * rotary_pairs` floats (cosines then sines).

    Returns q `(1, query_heads, 1, head_dim)` and k `(1, kv_heads, 1, head_dim)` — the
    layout attention consumes. `mscale` pre-scales the rotary inputs only; pass `1.0`
    for a plain rotary.
    """
    assert qk_norm_rope_decode_applies(head_dim, rotary_pairs)
    length = _check(
        raw_queries,
        raw_keys,
        query_weight,
        key_weight,
        angles,
        query_heads,
        kv_heads,
        head_dim,
        rotary_pairs,
    )
    assert length == 1
    assert angles.size == 2 * rotary_pairs
    out = _DECODE_KERNEL(
        inputs=[
            raw_queries.reshape(-1),
            raw_keys.reshape(-1),
            query_weight.reshape(-1),
            key_weight.reshape(-1),
            angles.reshape(-1),
            mx.array([eps], dtype=mx.float32),
            mx.array([mscale], dtype=mx.float32),
        ],
        template=[
            ("T", raw_queries.dtype),
            ("HEAD_DIM", head_dim),
            ("ROTARY_PAIRS", rotary_pairs),
            ("QUERY_HEADS", query_heads),
        ],
        grid=((query_heads + kv_heads) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, query_heads, 1, head_dim), (1, kv_heads, 1, head_dim)],
        output_dtypes=[raw_queries.dtype, raw_queries.dtype],
    )
    return out[0], out[1]


def qk_norm_rope_prefill_scaled_applies(head_dim: int, rotary_pairs: int) -> bool:
    """As the decode kernel, plus the segment's rows are threadgroups on the y axis."""
    return _lane_layout(head_dim, rotary_pairs)


def qk_norm_rope_prefill_scaled(
    raw_queries: mx.array,
    raw_keys: mx.array,
    *,
    query_weight: mx.array,
    key_weight: mx.array,
    angles: mx.array,
    offset: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_pairs: int,
    eps: float,
    mscale: float,
) -> tuple[mx.array, mx.array]:
    """A segment: `raw_queries` [length, query_heads, head_dim] and `raw_keys`
    [length, kv_heads, head_dim] (token-major, as the projection leaves them), normed
    per head and rotated by row `offset + t` of the `angles` table
    [positions, 2 * rotary_pairs].

    Returns q `(1, query_heads, length, head_dim)` and k `(1, kv_heads, length, head_dim)`
    — transposed to head-major on the way out, which is what attention consumes.
    `mscale` pre-scales the rotary inputs only; the non-rotating tail passes through.
    """
    assert qk_norm_rope_prefill_scaled_applies(head_dim, rotary_pairs)
    length = _check(
        raw_queries,
        raw_keys,
        query_weight,
        key_weight,
        angles,
        query_heads,
        kv_heads,
        head_dim,
        rotary_pairs,
    )
    assert angles.size >= (offset + length) * 2 * rotary_pairs
    out = _PREFILL_SCALED_KERNEL(
        inputs=[
            raw_queries.reshape(-1),
            raw_keys.reshape(-1),
            query_weight.reshape(-1),
            key_weight.reshape(-1),
            angles.reshape(-1),
            mx.array([offset], dtype=mx.int32),
            mx.array([eps], dtype=mx.float32),
            mx.array([mscale], dtype=mx.float32),
        ],
        template=[
            ("T", raw_queries.dtype),
            ("HEAD_DIM", head_dim),
            ("ROTARY_PAIRS", rotary_pairs),
            ("QUERY_HEADS", query_heads),
            ("KV_HEADS", kv_heads),
        ],
        grid=((query_heads + kv_heads) * 32, length, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, query_heads, length, head_dim), (1, kv_heads, length, head_dim)],
        output_dtypes=[raw_queries.dtype, raw_queries.dtype],
    )
    return out[0], out[1]


def qk_norm_rope_prefill_applies(head_dim: int, rotary_pairs: int) -> bool:
    """The unscaled prefill kernel has no tail branch: it only writes the rotary section,
    so the rotary section must be the whole head (`2 * rotary_pairs == head_dim`)."""
    return _lane_layout(head_dim, rotary_pairs) and 2 * rotary_pairs == head_dim


def qk_norm_rope_prefill(
    raw_queries: mx.array,
    raw_keys: mx.array,
    *,
    query_weight: mx.array,
    key_weight: mx.array,
    angles: mx.array,
    offset: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_pairs: int,
    eps: float,
) -> tuple[mx.array, mx.array]:
    """`qk_norm_rope_prefill_scaled` without the pre-scale, for the full-rotary case:
    every element of the head rotates, so there is no tail to copy and no mscale
    round-trip on the rotary inputs.
    """
    assert qk_norm_rope_prefill_applies(head_dim, rotary_pairs)
    length = _check(
        raw_queries,
        raw_keys,
        query_weight,
        key_weight,
        angles,
        query_heads,
        kv_heads,
        head_dim,
        rotary_pairs,
    )
    assert angles.size >= (offset + length) * 2 * rotary_pairs
    out = _PREFILL_KERNEL(
        inputs=[
            raw_queries.reshape(-1),
            raw_keys.reshape(-1),
            query_weight.reshape(-1),
            key_weight.reshape(-1),
            angles.reshape(-1),
            mx.array([offset], dtype=mx.int32),
            mx.array([eps], dtype=mx.float32),
        ],
        template=[
            ("T", raw_queries.dtype),
            ("HEAD_DIM", head_dim),
            ("ROTARY_PAIRS", rotary_pairs),
            ("QUERY_HEADS", query_heads),
            ("KV_HEADS", kv_heads),
        ],
        grid=((query_heads + kv_heads) * 32, length, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, query_heads, length, head_dim), (1, kv_heads, length, head_dim)],
        output_dtypes=[raw_queries.dtype, raw_queries.dtype],
    )
    return out[0], out[1]


@dataclass(frozen=True)
class QkNormRope(QkvRopeStrategy):
    """The table-driven prologue: per-head norm and rotary fused for a whole projection.

    Serves the declarations `RopeEpilogue` cannot — a partial rotary, a frequency table,
    a YaRN pre-scale — by taking the angles as an input instead of deriving them, which
    is also what keeps it exact: `angles(offset, length)` returns the very floats the op
    chain would have used, one row for a step and `length` contiguous rows for a segment.
    A segment needs an integer offset (the kernel indexes rows from it) and a single
    batch; anything else goes to the fallback.
    """

    projection: Projection
    heads: int
    kv_heads: int
    head_dim: int
    rotary_pairs: int
    eps: float
    mscale: float
    angles: Angles
    q_norm: mx.array
    k_norm: mx.array
    fallback: DefaultQkvRope

    @classmethod
    def build(
        cls,
        projection: Projection,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope: Rotation,
        eps: float,
        q_norm: mx.array | None,
        k_norm: mx.array | None,
        base: float | None,
        angles: Angles | None,
        rotary_pairs: int | None,
        mscale: float,
        segments: tuple[Leaf, Leaf, Leaf] | None,
    ) -> Self | None:
        if angles is None or rotary_pairs is None or q_norm is None or k_norm is None:
            return None
        if not qk_norm_rope_decode_applies(head_dim, rotary_pairs):
            return None
        return cls(
            projection,
            heads,
            kv_heads,
            head_dim,
            rotary_pairs,
            eps,
            mscale,
            angles,
            q_norm,
            k_norm,
            DefaultQkvRope.build(
                projection,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                rope=rope,
                eps=eps,
                q_norm=q_norm,
                k_norm=k_norm,
                base=base,
                angles=angles,
                rotary_pairs=rotary_pairs,
                mscale=mscale,
                segments=segments,
            ),
        )

    def _raw(self, fused: mx.array, length: int) -> tuple[mx.array, mx.array, mx.array]:
        """Token-major raw q and k — the layout the prefill kernels read — and the
        head-major values, which no kernel here touches."""
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        raw_queries, raw_keys, values = mx.split(
            fused, [query_width, query_width + kv_width], axis=-1
        )
        return (
            raw_queries,
            raw_keys,
            values.reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3),
        )

    def __call__(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        fused = self.projection(x)
        length = x.shape[1]
        if x.shape[0] != 1 or (length > 1 and not isinstance(offset, int)):
            return self.fallback.epilogue(fused, offset)
        raw_queries, raw_keys, values = self._raw(fused, length)
        angles = self.angles(offset, length)
        if length == 1:
            queries, keys = qk_norm_rope_decode(
                raw_queries,
                raw_keys,
                query_weight=self.q_norm,
                key_weight=self.k_norm,
                angles=angles,
                query_heads=self.heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                rotary_pairs=self.rotary_pairs,
                eps=self.eps,
                mscale=self.mscale,
            )
            return queries, keys, values
        if self.mscale == 1.0 and qk_norm_rope_prefill_applies(
            self.head_dim, self.rotary_pairs
        ):
            queries, keys = qk_norm_rope_prefill(
                raw_queries,
                raw_keys,
                query_weight=self.q_norm,
                key_weight=self.k_norm,
                angles=angles,
                offset=0,
                query_heads=self.heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                rotary_pairs=self.rotary_pairs,
                eps=self.eps,
            )
            return queries, keys, values
        queries, keys = qk_norm_rope_prefill_scaled(
            raw_queries,
            raw_keys,
            query_weight=self.q_norm,
            key_weight=self.k_norm,
            angles=angles,
            offset=0,
            query_heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            rotary_pairs=self.rotary_pairs,
            eps=self.eps,
            mscale=self.mscale,
        )
        return queries, keys, values
