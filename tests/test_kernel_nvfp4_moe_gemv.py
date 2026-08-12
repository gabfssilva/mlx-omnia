"""Parity for the NVFP4 routed-expert decode kernel against `gather_qmm`'s own arithmetic.

The kernel reproduces mlx's nvfp4 dequantization (e2m1 nibbles, one e4m3 byte per group of
16) rather than reusing it, so the comparison that matters is against `mx.gather_qmm(...,
mode="nvfp4")` over the same packed tensors: any slip in the scale's exponent bias is a
factor of two, not a rounding difference, and shows up far above the fp32 floor.
"""

import mlx.core as mx
import pytest
from conftest import relative_diff

from mlx_omnia.engine.core.kernels.down_combine.nvfp4 import Nvfp4DownCombine
from mlx_omnia.engine.core.kernels.gate_up.nvfp4 import Nvfp4GateUp
from mlx_omnia.engine.core.kernels.gate_up.nvfp4 import applies as nvfp4_moe_applies

EXPERTS = 8
ACTIVE = 4
HIDDEN = 64
INNER = 32
GROUP = 16
BITS = 4


def quantized(shape: tuple[int, ...], seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    dense = mx.random.normal(shape).astype(mx.float32)
    packed, scales, *_ = mx.quantize(dense, group_size=GROUP, bits=BITS, mode="nvfp4")
    mx.eval(packed, scales)
    return packed, scales


def reference(
    x: mx.array, weight: mx.array, scales: mx.array, indices: mx.array
) -> mx.array:
    return mx.gather_qmm(
        x,
        weight,
        scales,
        None,
        rhs_indices=indices[None],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    )


def assert_matches(ours: mx.array, expected: mx.array) -> None:
    """A scale plane of zeros makes both sides identically zero, and the house metric
    divides by the reference's magnitude — there is no relative bound to take there."""
    if float(mx.max(mx.abs(expected))) == 0.0:
        assert float(mx.max(mx.abs(ours))) == 0.0
        return
    assert relative_diff(ours, expected) < 1e-5


def gate_up_reference(
    x: mx.array, weight: mx.array, scales: mx.array, indices: mx.array
) -> mx.array:
    projected = reference(x[None, None, None], weight, scales, indices)
    pairs = projected.reshape(ACTIVE, INNER, 2)
    gate = pairs[..., 0]
    return gate * mx.sigmoid(gate) * pairs[..., 1]


def test_applies_requires_whole_groups() -> None:
    assert nvfp4_moe_applies(HIDDEN, INNER)
    assert not nvfp4_moe_applies(HIDDEN + 8, INNER)
    assert not nvfp4_moe_applies(HIDDEN, INNER + 8)


def test_gate_up_matches_gather_qmm() -> None:
    weight, scales = quantized((EXPERTS, 2 * INNER, HIDDEN), seed=0)
    x = mx.random.normal((HIDDEN,)).astype(mx.float32)
    indices = mx.array([1, 3, 4, 6], dtype=mx.uint32)

    fused = Nvfp4GateUp(weight, scales)(x, indices)

    assert_matches(fused.reshape(ACTIVE, INNER), gate_up_reference(x, weight, scales, indices))


def test_down_combine_matches_gather_qmm() -> None:
    weight, scales = quantized((EXPERTS, HIDDEN, INNER), seed=1)
    act = mx.random.normal((ACTIVE, INNER)).astype(mx.float32)
    indices = mx.array([0, 2, 5, 7], dtype=mx.uint32)
    routing = mx.random.uniform(shape=(ACTIVE,)).astype(mx.float32)
    residual = mx.random.normal((HIDDEN,)).astype(mx.float32)

    fused = Nvfp4DownCombine(weight, scales)(act, indices, routing, residual)

    projected = reference(act[None, :, None], weight, scales, indices).squeeze((0, 2))
    expected = (projected * routing[:, None]).sum(axis=0) + residual
    assert_matches(fused.reshape(HIDDEN), expected)


@pytest.mark.parametrize("code", [0x00, 0x01, 0x07, 0x38, 0x7E, 0x80, 0xFF])
def test_scale_codes_match_dequantize(code: int) -> None:
    """`mx.quantize` never emits a denormal or a saturated scale, so the e4m3 decode's
    edges are only reachable by writing the scale plane directly."""
    weight, scales = quantized((EXPERTS, 2 * INNER, HIDDEN), seed=2)
    scales = mx.full(scales.shape, code, dtype=mx.uint8)
    x = mx.random.normal((HIDDEN,)).astype(mx.float32)
    indices = mx.array([1, 3, 4, 6], dtype=mx.uint32)

    fused = Nvfp4GateUp(weight, scales)(x, indices)

    assert_matches(fused.reshape(ACTIVE, INNER), gate_up_reference(x, weight, scales, indices))
