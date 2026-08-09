"""Parity for the fused down/route/residual kernel against `gather_qmm` compositions.

The kernel is one dispatch for what is otherwise a routed `gather_qmm`, an unrouted
`quantized_matmul`, a weighted sum over experts and an add, so the reference is exactly
that composition — with the bf16 epilogue reproduced in the kernel's order, since the
routed total accumulates slot by slot and each product rounds before it lands.
"""

import mlx.core as mx
from conftest import relative_diff

from sideros.core.kernels.nvfp4_fused_down_residual import (
    halve_down_scales,
    nvfp4_fused_down_residual,
    nvfp4_fused_down_residual_applies,
)

HIDDEN = 64
INNER = 512
EXPERTS = 8
TOPK = 4
GROUP = 16
BITS = 4
SCALING = 2.5
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


def down_reference(
    act: mx.array, weight: mx.array, scales: mx.array, indices: mx.array
) -> mx.array:
    return mx.gather_qmm(
        act.astype(mx.float32)[None, :, None],
        weight,
        scales,
        None,
        rhs_indices=indices[None],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    ).reshape(indices.size, HIDDEN)


def shared_reference(act: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.gather_qmm(
        act.astype(mx.float32)[None, None, None],
        weight[None],
        scales[None],
        None,
        rhs_indices=mx.array([[0]], dtype=mx.uint32),
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    ).reshape(HIDDEN)


def combine_reference(
    routed: mx.array, shared: mx.array, routing: mx.array, residual: mx.array
) -> mx.array:
    weights = routing.astype(mx.bfloat16)
    total = mx.zeros((HIDDEN,), dtype=mx.bfloat16)
    for slot in range(TOPK):
        total = routed[slot] * weights[slot] + total
    scaled = total * mx.array(SCALING, dtype=mx.bfloat16)
    return residual + (scaled + shared)


def test_applies_pins_the_unlooped_contraction() -> None:
    assert nvfp4_fused_down_residual_applies(HIDDEN, INNER, TOPK)
    assert not nvfp4_fused_down_residual_applies(HIDDEN, INNER // 2, TOPK)
    assert not nvfp4_fused_down_residual_applies(HIDDEN + 1, INNER, TOPK)
    assert not nvfp4_fused_down_residual_applies(HIDDEN, INNER, 32)


def test_matches_gather_qmm_composition() -> None:
    routed_weight, routed_scales = quantized_paired((EXPERTS, HIDDEN, INNER), seed=0)
    shared_weight, shared_scales = quantized_paired((HIDDEN, INNER), seed=1)
    routed_halved = halve_down_scales(routed_scales)
    shared_halved = halve_down_scales(shared_scales)
    assert routed_halved is not None and shared_halved is not None

    act = mx.random.normal((TOPK, INNER)).astype(mx.bfloat16)
    shared_act = mx.random.normal((INNER,)).astype(mx.bfloat16)
    indices = mx.array([0, 2, 5, 7], dtype=mx.uint32)
    routing = mx.random.uniform(shape=(TOPK,)).astype(mx.float32)
    residual = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    fused = nvfp4_fused_down_residual(
        act,
        routed_weight,
        routed_halved,
        indices,
        routing,
        shared_act,
        shared_weight,
        shared_halved,
        residual,
        SCALING,
    )

    routed = down_reference(act, routed_weight, routed_scales, indices).astype(mx.bfloat16)
    shared = shared_reference(shared_act, shared_weight, shared_scales).astype(mx.bfloat16)
    assert_matches(fused, combine_reference(routed, shared, routing, residual))


def test_patch_header_restores_the_first_span() -> None:
    """Row 0 of the first expert is the one span mlx's quantizer writes twice; its odd byte
    lives in the header, for the routed plane and the unrouted one alike."""
    routed_weight, routed_scales = quantized_paired((EXPERTS, HIDDEN, INNER), seed=2)
    shared_weight, shared_scales = quantized_paired((HIDDEN, INNER), seed=3)
    routed_scales[0, 0, 1] = mx.array(0x40, dtype=mx.uint8)
    shared_scales[0, 1] = mx.array(0x41, dtype=mx.uint8)
    routed_halved = halve_down_scales(routed_scales)
    shared_halved = halve_down_scales(shared_scales)
    assert routed_halved is not None and shared_halved is not None

    act = mx.random.normal((TOPK, INNER)).astype(mx.bfloat16)
    shared_act = mx.random.normal((INNER,)).astype(mx.bfloat16)
    # Expert 0 in a slot, so the patched routed row is actually read.
    indices = mx.array([3, 0, 5, 7], dtype=mx.uint32)
    routing = mx.random.uniform(shape=(TOPK,)).astype(mx.float32)
    residual = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    fused = nvfp4_fused_down_residual(
        act,
        routed_weight,
        routed_halved,
        indices,
        routing,
        shared_act,
        shared_weight,
        shared_halved,
        residual,
        SCALING,
    )

    routed = down_reference(act, routed_weight, routed_scales, indices).astype(mx.bfloat16)
    shared = shared_reference(shared_act, shared_weight, shared_scales).astype(mx.bfloat16)
    assert_matches(fused, combine_reference(routed, shared, routing, residual))
