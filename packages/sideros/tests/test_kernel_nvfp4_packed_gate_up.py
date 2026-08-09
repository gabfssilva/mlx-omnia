"""Parity for the packed routed gate/up kernel against `gather_qmm`'s own arithmetic.

Three things can only be wrong together with the reference: the bit-trick NVFP4 decode
(off by a power of two, never by a rounding), the walk-order scale bank (a scale byte read
one k-block or one sub away is a different group's), and the inline tournament (the wrong
expert entirely). So the comparison is against `mx.gather_qmm(..., mode="nvfp4")` over the
same tensors, with the expert set computed independently from the ordinal keys.

The kernel rounds its fp32 accumulator to bf16 once and then runs a bf16 SwiGLU chain; the
reference reproduces that chain expression for expression, so the only slack is the fp32
accumulation order upstream of the single rounding — one bf16 ulp of it, when a dot lands
near a rounding boundary.
"""

import random

import mlx.core as mx
from conftest import relative_diff

from sideros.core.kernels.nvfp4_packed import halved_group32_scales
from sideros.core.kernels.nvfp4_packed_gate_up import (
    nvfp4_packed_gate_up,
    nvfp4_packed_gate_up_applies,
    pack_gate_up_scales,
)

HIDDEN = 512
INNER = 32
EXPERTS = 64
TOPK = 4
GROUP = 16
BITS = 4
SLACK = 2.0**-6


def quantized_paired(shape: tuple[int, ...], seed: int) -> tuple[mx.array, mx.array]:
    """An NVFP4 checkpoint whose adjacent group pairs share a scale byte — what mlx's
    Metal `fp_quantize` produces, and the only shape the halved plane may carry."""
    mx.random.seed(seed)
    dense = mx.random.normal(shape).astype(mx.float32)
    packed, scales, *_ = mx.quantize(dense, group_size=GROUP, bits=BITS, mode="nvfp4")
    paired = mx.repeat(scales[..., ::2], 2, axis=-1)
    mx.eval(packed, paired)
    return packed, paired


def assert_matches(ours: mx.array, expected: mx.array) -> None:
    """A scale plane of zeros makes both sides identically zero, and the house metric
    divides by the reference's magnitude — there is no relative bound to take there."""
    if float(mx.max(mx.abs(expected))) == 0.0:
        assert float(mx.max(mx.abs(ours))) == 0.0
        return
    assert relative_diff(ours, expected) < SLACK


def gate_rows() -> list[int]:
    """The fused bank interleaves gate and up in tiles of 32 rows."""
    return [(row // 32) * 64 + row % 32 for row in range(INNER)]


def swiglu_reference(
    x: mx.array, weight: mx.array, scales: mx.array, experts: list[int]
) -> mx.array:
    projected = mx.gather_qmm(
        x.astype(mx.float32)[None, None, None],
        weight,
        scales,
        None,
        rhs_indices=mx.array(experts, dtype=mx.uint32)[None],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    ).reshape(len(experts), 2 * INNER)
    gate = mx.take(projected, mx.array(gate_rows()), axis=1).astype(mx.bfloat16)
    up = mx.take(projected, mx.array([r + 32 for r in gate_rows()]), axis=1).astype(
        mx.bfloat16
    )
    one = mx.array(1, dtype=mx.bfloat16)
    y = one / (one + mx.exp(mx.abs(gate)))
    sigmoid = mx.where(gate < 0, y, one - y)
    return (gate * sigmoid) * up


def selected(keys: list[int]) -> list[int]:
    """Ascending ordinal, ties to the lower expert index — the tournament's rule."""
    return sorted(range(len(keys)), key=lambda e: (keys[e], e))[:TOPK]


def shuffled_keys(seed: int) -> list[int]:
    """Distinct ordinals in a fixed shuffled order; `0` is below every one of them."""
    order = random.Random(seed).sample(range(EXPERTS), EXPERTS)
    return [value * 1000 + 5 for value in order]


def test_applies_requires_blocked_contraction() -> None:
    assert nvfp4_packed_gate_up_applies(HIDDEN, INNER, EXPERTS, TOPK)
    assert not nvfp4_packed_gate_up_applies(HIDDEN + 16, INNER, EXPERTS, TOPK)
    assert not nvfp4_packed_gate_up_applies(HIDDEN, INNER + 16, EXPERTS, TOPK)
    assert not nvfp4_packed_gate_up_applies(HIDDEN, INNER, EXPERTS + 16, TOPK)
    assert not nvfp4_packed_gate_up_applies(HIDDEN, INNER, EXPERTS, EXPERTS + 1)


def test_matches_gather_qmm() -> None:
    weight, scales = quantized_paired((EXPERTS, 2 * INNER, HIDDEN), seed=0)
    packed_scales = pack_gate_up_scales(scales)
    assert packed_scales is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    keys = shuffled_keys(seed=1)

    fused = nvfp4_packed_gate_up(
        x, weight, packed_scales, mx.array(keys, dtype=mx.uint32), TOPK
    )

    assert_matches(fused, swiglu_reference(x, weight, scales, selected(keys)))


def test_ties_resolve_to_the_lower_expert_index() -> None:
    weight, scales = quantized_paired((EXPERTS, 2 * INNER, HIDDEN), seed=2)
    packed_scales = pack_gate_up_scales(scales)
    assert packed_scales is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    keys = shuffled_keys(seed=3)
    # Two experts share the best ordinal: the lower index must take the first slot.
    keys[5] = 0
    keys[9] = 0
    assert selected(keys)[:2] == [5, 9]

    fused = nvfp4_packed_gate_up(
        x, weight, packed_scales, mx.array(keys, dtype=mx.uint32), TOPK
    )

    assert_matches(fused, swiglu_reference(x, weight, scales, selected(keys)))


def test_patch_header_restores_the_two_unequal_spans() -> None:
    """The quantizer's first simdgroup writes the first span of a tensor twice, so expert
    0's gate row 0 and up row 0 may break the pairing. Their odd byte lives in the header
    and one lane per row reads it from there."""
    weight, scales = quantized_paired((EXPERTS, 2 * INNER, HIDDEN), seed=4)
    scales[0, 0, 1] = mx.array(0x40, dtype=mx.uint8)
    scales[0, 32, 1] = mx.array(0x41, dtype=mx.uint8)
    packed_scales = pack_gate_up_scales(scales)
    assert packed_scales is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    keys = shuffled_keys(seed=5)
    keys[0] = 0  # expert 0 selected, so the patched rows are actually read

    fused = nvfp4_packed_gate_up(
        x, weight, packed_scales, mx.array(keys, dtype=mx.uint32), TOPK
    )

    assert_matches(fused, swiglu_reference(x, weight, scales, selected(keys)))


def test_packing_refuses_an_unpaired_plane() -> None:
    _, scales = quantized_paired((EXPERTS, 2 * INNER, HIDDEN), seed=6)
    scales[1, 3, 5] = mx.array(0x42, dtype=mx.uint8)
    assert pack_gate_up_scales(scales) is None


def test_packing_refuses_a_signed_scale_byte() -> None:
    """A byte with its sign bit set decodes through the wrong exponent, not as a negative
    number: the halved plane is only installable while the whole plane stays positive."""
    _, scales = quantized_paired((EXPERTS, 2 * INNER, HIDDEN), seed=7)
    scales[2, 4, 6] = mx.array(0x80, dtype=mx.uint8)
    scales[2, 4, 7] = mx.array(0x80, dtype=mx.uint8)
    assert halved_group32_scales(scales, (0,)) is None
