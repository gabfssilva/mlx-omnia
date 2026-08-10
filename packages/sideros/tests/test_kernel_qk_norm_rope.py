"""qk_norm_rope against the op composition it fuses: `mx.fast.rms_norm` per head, then
the rotary applied by hand with the *same* `angles` table.

The reference is written out rather than taken from `mx.fast.rope` on purpose: the
kernels do not derive frequencies, they read a precomputed table, and a table-driven
rotary is exactly `first * cos - second * sin` over the half-split pairs of the rotary
section. Reproducing it with mx ops keeps the comparison about what the kernel fuses
(the norm, the pre-scale round-trip, the partial-rotary boundary) instead of about how
two implementations invent their angles.

Floors: the fp32 template is compared at 1e-5 — the rounding differences are two fp32
ops deep. bf16 is compared at two bf16 ulps (2 * 2^-8): the norm is bit-exact (same
`per_lane`-in-index-order + `simd_sum` reduction `rms_single_row` runs at axis 128) and
the rotation accumulates in fp32 and rounds once on both sides, so the only free
difference is the fma contraction the Metal compiler may apply to
`first * cos - second * sin`, worth at most one ulp of the bf16 result.
"""

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from conftest import relative_diff

from sideros.core.kernels.qkv_rope.qk_norm_rope import (
    _OUTPUTS,
    _PREFILL_SCALED_INPUTS,
    _PREFILL_SCALED_SOURCE,
    qk_norm_rope_decode,
    qk_norm_rope_decode_applies,
    qk_norm_rope_prefill,
    qk_norm_rope_prefill_applies,
    qk_norm_rope_prefill_scaled,
    qk_norm_rope_prefill_scaled_applies,
)
from sideros.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from sideros.core.mxcompat import MetalKernel

QUERY_HEADS = 12
KV_HEADS = 4
HEAD_DIM = 128
LENGTH = 5
OFFSET = 3
EPS = 1e-6
MSCALE = 1.3465735912322998
BF16_ULP = 2.0**-8


def _angles(positions: int, rotary_pairs: int, base: float, first: int = 0) -> mx.array:
    """Rows `first .. first + positions` of the table, `[positions, 2 * rotary_pairs]`:
    cosines then sines, the layout both kernels read.

    `first` exists so a caller that needs a single non-zero position gets its own
    freshly allocated array rather than a slice of a taller table. Position 0 is the
    identity rotation (every theta is 0, so cos is 1 and sin is 0), which makes a test
    at that row vacuous; slicing row `first` out of a taller table fixes the vacuity but
    hands the kernel a row-contiguous *view with a non-zero data offset*, which
    `ensure_row_contiguous` does not copy (`mx.contiguous` would not either — the flag
    is already set). Building the row directly keeps both hazards away.
    """
    inverse = np.exp(-np.log(base) * np.arange(rotary_pairs) / rotary_pairs)
    theta = (first + np.arange(positions))[:, None] * inverse[None, :]
    table = np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)
    return mx.array(table.astype(np.float32))


def _inputs(
    dtype: mx.Dtype, length: int, head_dim: int, seed: int
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Raw q/k straight off a projection, plus norm weights scattered around one."""
    rng = np.random.default_rng(seed)
    raw_queries = mx.array(
        rng.standard_normal((length, QUERY_HEADS, head_dim)).astype(np.float32)
    ).astype(dtype)
    raw_keys = mx.array(
        rng.standard_normal((length, KV_HEADS, head_dim)).astype(np.float32)
    ).astype(dtype)
    query_weight = mx.array((1.0 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32)).astype(
        dtype
    )
    key_weight = mx.array((1.0 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32)).astype(
        dtype
    )
    return raw_queries, raw_keys, query_weight, key_weight


def _normed(raw: mx.array, weight: mx.array, heads: int, head_dim: int) -> mx.array:
    """`[length, heads, head_dim]` -> `[1, heads, length, head_dim]`, norm only."""
    normed = mx.fast.rms_norm(raw.reshape(-1, heads, head_dim), weight, EPS)
    return normed.transpose(1, 0, 2)[None]


def _reference(
    raw: mx.array,
    weight: mx.array,
    heads: int,
    head_dim: int,
    rotary_pairs: int,
    rows: mx.array,
    mscale: float,
) -> mx.array:
    """The stock composition, `[1, heads, length, head_dim]`.

    `rows` is the `[length, 2 * rotary_pairs]` slice of the angle table the segment uses.
    The `mscale` round-trip is reproduced as the kernel writes it — `float(T(x * T(m)))`,
    so the factor is rounded to T before the multiply and the product rounded again —
    and it touches the rotary inputs only, never the tail.
    """
    dtype = raw.dtype
    normed = mx.fast.rms_norm(raw.reshape(-1, heads, head_dim), weight, EPS)
    cosine = rows[:, None, :rotary_pairs].astype(mx.float32)
    sine = rows[:, None, rotary_pairs:].astype(mx.float32)
    # float32 first, then T: the kernel reads an fp32 scalar input and rounds it in-kernel,
    # so rounding the Python double straight to T would be a different number.
    scale = mx.array([mscale], dtype=mx.float32).astype(dtype).astype(mx.float32)
    wide = normed.astype(mx.float32)
    first = (wide[..., :rotary_pairs] * scale).astype(dtype).astype(mx.float32)
    second = (wide[..., rotary_pairs : 2 * rotary_pairs] * scale).astype(dtype).astype(mx.float32)
    rotated = mx.concatenate(
        [first * cosine - second * sine, first * sine + second * cosine], axis=-1
    ).astype(dtype)
    tail = normed[..., 2 * rotary_pairs :]
    out = rotated if 2 * rotary_pairs == head_dim else mx.concatenate([rotated, tail], axis=-1)
    return out.transpose(1, 0, 2)[None]


SHAPES = ((128, 32), (128, 64), (64, 16))


def test_decode_matches_ops_fp32() -> None:
    for head_dim, rotary_pairs in SHAPES:
        raw_q, raw_k, q_weight, k_weight = _inputs(mx.float32, 1, head_dim, seed=7)
        angles = _angles(1, rotary_pairs, base=1_000_000.0, first=OFFSET)
        q, k = qk_norm_rope_decode(
            raw_q,
            raw_k,
            query_weight=q_weight,
            key_weight=k_weight,
            angles=angles,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=head_dim,
            rotary_pairs=rotary_pairs,
            eps=EPS,
            mscale=MSCALE,
        )
        args = (head_dim, rotary_pairs, angles, MSCALE)
        assert relative_diff(q, _reference(raw_q, q_weight, QUERY_HEADS, *args)) < 1e-5
        assert relative_diff(k, _reference(raw_k, k_weight, KV_HEADS, *args)) < 1e-5


def test_decode_matches_ops_bf16() -> None:
    for head_dim, rotary_pairs in SHAPES:
        raw_q, raw_k, q_weight, k_weight = _inputs(mx.bfloat16, 1, head_dim, seed=7)
        angles = _angles(1, rotary_pairs, base=1_000_000.0, first=OFFSET)
        q, k = qk_norm_rope_decode(
            raw_q,
            raw_k,
            query_weight=q_weight,
            key_weight=k_weight,
            angles=angles,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=head_dim,
            rotary_pairs=rotary_pairs,
            eps=EPS,
            mscale=MSCALE,
        )
        args = (head_dim, rotary_pairs, angles, MSCALE)
        assert relative_diff(q, _reference(raw_q, q_weight, QUERY_HEADS, *args)) < 2 * BF16_ULP
        assert relative_diff(k, _reference(raw_k, k_weight, KV_HEADS, *args)) < 2 * BF16_ULP


def _prefill_scaled(
    raw_q: mx.array,
    raw_k: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
    head_dim: int,
    rotary_pairs: int,
) -> tuple[mx.array, mx.array]:
    return qk_norm_rope_prefill_scaled(
        raw_q,
        raw_k,
        query_weight=q_weight,
        key_weight=k_weight,
        angles=angles,
        offset=OFFSET,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=head_dim,
        rotary_pairs=rotary_pairs,
        eps=EPS,
        mscale=MSCALE,
    )


def _prefill_case(dtype: mx.Dtype, head_dim: int, rotary_pairs: int, tolerance: float) -> None:
    raw_q, raw_k, q_weight, k_weight = _inputs(dtype, LENGTH, head_dim, seed=13)
    angles = _angles(OFFSET + LENGTH, rotary_pairs, base=1_000_000.0)
    rows = angles[OFFSET : OFFSET + LENGTH]
    q, k = _prefill_scaled(raw_q, raw_k, q_weight, k_weight, angles, head_dim, rotary_pairs)
    args = (head_dim, rotary_pairs, rows, MSCALE)
    assert relative_diff(q, _reference(raw_q, q_weight, QUERY_HEADS, *args)) < tolerance
    assert relative_diff(k, _reference(raw_k, k_weight, KV_HEADS, *args)) < tolerance


def test_prefill_scaled_matches_ops_fp32() -> None:
    for head_dim, rotary_pairs in SHAPES:
        _prefill_case(mx.float32, head_dim, rotary_pairs, 1e-5)


def test_prefill_scaled_matches_ops_bf16() -> None:
    for head_dim, rotary_pairs in SHAPES:
        _prefill_case(mx.bfloat16, head_dim, rotary_pairs, 2 * BF16_ULP)


def _prefill_plain_case(dtype: mx.Dtype, head_dim: int, tolerance: float) -> None:
    """The unscaled kernel is full-rotary only: `2 * rotary_pairs == head_dim`."""
    rotary_pairs = head_dim // 2
    raw_q, raw_k, q_weight, k_weight = _inputs(dtype, LENGTH, head_dim, seed=21)
    angles = _angles(OFFSET + LENGTH, rotary_pairs, base=10_000.0)
    q, k = qk_norm_rope_prefill(
        raw_q,
        raw_k,
        query_weight=q_weight,
        key_weight=k_weight,
        angles=angles,
        offset=OFFSET,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=head_dim,
        rotary_pairs=rotary_pairs,
        eps=EPS,
    )
    rows = angles[OFFSET : OFFSET + LENGTH]
    args = (head_dim, rotary_pairs, rows, 1.0)
    assert relative_diff(q, _reference(raw_q, q_weight, QUERY_HEADS, *args)) < tolerance
    assert relative_diff(k, _reference(raw_k, k_weight, KV_HEADS, *args)) < tolerance


def test_prefill_matches_ops_fp32() -> None:
    for head_dim in (128, 64):
        _prefill_plain_case(mx.float32, head_dim, 1e-5)


def test_prefill_matches_ops_bf16() -> None:
    for head_dim in (128, 64):
        _prefill_plain_case(mx.bfloat16, head_dim, 2 * BF16_ULP)


ROTARY_PAIRS = 32


def test_partial_rotary_tail_is_the_norm_verbatim() -> None:
    """Elements `[2 * rotary_pairs, head_dim)` are the rms_norm output, bit for bit —
    not rotated, not scaled by mscale, not re-rounded. Asserted in bf16, where the
    kernel's reduction is `rms_single_row`'s at axis 128 and equality is the real claim.
    """
    raw_q, raw_k, q_weight, k_weight = _inputs(mx.bfloat16, LENGTH, HEAD_DIM, seed=13)
    angles = _angles(OFFSET + LENGTH, ROTARY_PAIRS, base=1_000_000.0)
    q, k = _prefill_scaled(raw_q, raw_k, q_weight, k_weight, angles, HEAD_DIM, ROTARY_PAIRS)
    boundary = 2 * ROTARY_PAIRS
    for out, raw, weight, heads in (
        (q, raw_q, q_weight, QUERY_HEADS),
        (k, raw_k, k_weight, KV_HEADS),
    ):
        normed = _normed(raw, weight, heads, HEAD_DIM)
        assert bool(mx.array_equal(out[..., boundary:], normed[..., boundary:]))
        # The last rotating element did move, so the boundary is at `boundary`, not before it.
        assert not bool(mx.array_equal(out[..., boundary - 1], normed[..., boundary - 1]))
        assert bool(mx.array_equal(out[..., boundary], normed[..., boundary]))


def test_partial_rotary_tail_ignores_the_angles() -> None:
    """Same inputs, a different angle table: the rotary section moves, the tail does not
    change by a single bit. Independent of the norm reference, in both dtypes."""
    boundary = 2 * ROTARY_PAIRS
    for dtype in (mx.float32, mx.bfloat16):
        raw_q, raw_k, q_weight, k_weight = _inputs(dtype, LENGTH, HEAD_DIM, seed=13)
        first = _angles(OFFSET + LENGTH, ROTARY_PAIRS, base=1_000_000.0)
        second = _angles(OFFSET + LENGTH, ROTARY_PAIRS, base=10_000.0)
        qa, ka = _prefill_scaled(raw_q, raw_k, q_weight, k_weight, first, HEAD_DIM, ROTARY_PAIRS)
        qb, kb = _prefill_scaled(raw_q, raw_k, q_weight, k_weight, second, HEAD_DIM, ROTARY_PAIRS)
        for a, b in ((qa, qb), (ka, kb)):
            assert bool(mx.array_equal(a[..., boundary:], b[..., boundary:]))
            assert not bool(mx.array_equal(a[..., :boundary], b[..., :boundary]))


def test_applies_contracts() -> None:
    assert qk_norm_rope_decode_applies(128, 32)
    assert qk_norm_rope_decode_applies(128, 64)
    assert qk_norm_rope_decode_applies(64, 16)
    # 2 * rotary_pairs must fit in the head.
    assert not qk_norm_rope_decode_applies(128, 128)
    # 48 pairs is 12 lanes at head_dim 128: not a power of two, so `lane ^ n` is not the
    # rotary partner.
    assert not qk_norm_rope_decode_applies(128, 48)
    # The head must tile evenly over exactly 32 lanes.
    assert not qk_norm_rope_decode_applies(100, 32)
    assert qk_norm_rope_prefill_scaled_applies(128, 32)
    # The unscaled kernel writes no tail, so it is full-rotary only.
    assert not qk_norm_rope_prefill_applies(128, 32)
    assert qk_norm_rope_prefill_applies(128, 64)


def _dispatch(
    kernel: "MetalKernel",
    raw_q: mx.array,
    raw_k: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    angles: mx.array,
) -> tuple[mx.array, mx.array]:
    out = kernel(
        inputs=[
            raw_q.reshape(-1),
            raw_k.reshape(-1),
            q_weight.reshape(-1),
            k_weight.reshape(-1),
            angles.reshape(-1),
            mx.array([OFFSET], dtype=mx.int32),
            mx.array([EPS], dtype=mx.float32),
            mx.array([MSCALE], dtype=mx.float32),
        ],
        template=[
            ("T", raw_q.dtype),
            ("HEAD_DIM", HEAD_DIM),
            ("ROTARY_PAIRS", ROTARY_PAIRS),
            ("QUERY_HEADS", QUERY_HEADS),
            ("KV_HEADS", KV_HEADS),
        ],
        grid=((QUERY_HEADS + KV_HEADS) * 32, LENGTH, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, QUERY_HEADS, LENGTH, HEAD_DIM),
            (1, KV_HEADS, LENGTH, HEAD_DIM),
        ],
        output_dtypes=[raw_q.dtype, raw_q.dtype],
    )
    return out[0], out[1]


MUTATIONS = {
    # The single easiest thing to get subtly wrong: half the rotary section stops
    # rotating and joins the pass-through tail.
    "rotary_section_halved": (
        "constexpr uint rotary_pairs = ROTARY_PAIRS;",
        "constexpr uint rotary_pairs = ROTARY_PAIRS / 2;",
    ),
    "rotation_sign": (
        "output[pair] = T(first * cosine - second * sine);",
        "output[pair] = T(first * cosine + second * sine);",
    ),
    "cosine_reads_the_sine": (
        "float cosine = angle_row[pair];",
        "float cosine = angle_row[pair + rotary_pairs];",
    ),
    "wrong_rotary_partner": (
        "paired[i] = simd_shuffle(float(normalized[i]), lane ^ rotary_lanes);",
        "paired[i] = simd_shuffle(float(normalized[i]), lane ^ (2 * rotary_lanes));",
    ),
    "mscale_dropped": (
        "T rounded_mscale = T(mscale[0]);",
        "T rounded_mscale = T(1.0f);",
    ),
    "position_frozen": (
        "angles + (uint(offsets[0]) + t) * (2 * rotary_pairs);",
        "angles + uint(offsets[0]) * (2 * rotary_pairs);",
    ),
}


def test_mutations_are_caught() -> None:
    raw_q, raw_k, q_weight, k_weight = _inputs(mx.bfloat16, LENGTH, HEAD_DIM, seed=13)
    angles = _angles(OFFSET + LENGTH, ROTARY_PAIRS, base=1_000_000.0)
    rows = angles[OFFSET : OFFSET + LENGTH]
    args = (HEAD_DIM, ROTARY_PAIRS, rows, MSCALE)
    reference_q = _reference(raw_q, q_weight, QUERY_HEADS, *args)
    reference_k = _reference(raw_k, k_weight, KV_HEADS, *args)
    for name, (old, new) in MUTATIONS.items():
        assert old in _PREFILL_SCALED_SOURCE, name
        broken = metal_kernel(
            name=f"qk_norm_rope_prefill_scaled_{name}",
            input_names=_PREFILL_SCALED_INPUTS,
            output_names=_OUTPUTS,
            source=_PREFILL_SCALED_SOURCE.replace(old, new),
        )
        q, k = _dispatch(broken, raw_q, raw_k, q_weight, k_weight, angles)
        worst = max(relative_diff(q, reference_q), relative_diff(k, reference_k))
        assert worst > 2 * BF16_ULP, name
