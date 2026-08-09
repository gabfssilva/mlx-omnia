"""Parity for the unrouted halved-scale gate/up kernel against `gather_qmm`.

The kernel decodes NVFP4 with bit shuffling instead of a lookup — every value comes out
`2^-22` small and one multiply at the bf16 boundary puts it back — and reads one scale byte
per 32 weights instead of per 16. Both are exactness claims about mlx's own format, so the
reference is `mx.gather_qmm(..., mode="nvfp4")` over a one-expert bank holding the same
bytes, with the SwiGLU chain reproduced expression for expression in bf16.
"""

import mlx.core as mx
import pytest
from conftest import relative_diff

from sideros.core.kernels.nvfp4_halved_gate_up import (
    halve_gate_up_scales,
    nvfp4_halved_gate_up,
    nvfp4_halved_gate_up_applies,
)

HIDDEN = 512
INNER = 32
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


def swiglu_reference(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    projected = mx.gather_qmm(
        x.astype(mx.float32)[None, None, None],
        weight[None],
        scales[None],
        None,
        rhs_indices=mx.array([[0]], dtype=mx.uint32),
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    ).reshape(2 * INNER)
    gate = projected[:INNER].astype(mx.bfloat16)
    up = projected[INNER:].astype(mx.bfloat16)
    one = mx.array(1, dtype=mx.bfloat16)
    y = one / (one + mx.exp(mx.abs(gate)))
    sigmoid = mx.where(gate < 0, y, one - y)
    return (gate * sigmoid) * up


def test_applies_requires_blocked_contraction() -> None:
    assert nvfp4_halved_gate_up_applies(HIDDEN, INNER)
    assert not nvfp4_halved_gate_up_applies(HIDDEN + 16, INNER)
    assert not nvfp4_halved_gate_up_applies(HIDDEN, INNER + 1)


def test_matches_gather_qmm() -> None:
    weight, scales = quantized_paired((2 * INNER, HIDDEN), seed=0)
    halved = halve_gate_up_scales(scales)
    assert halved is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    fused = nvfp4_halved_gate_up(x, weight, halved)

    assert_matches(fused, swiglu_reference(x, weight, scales))


def test_patch_header_restores_the_two_unequal_spans() -> None:
    """Gate row 0 and up row 0 are the spans mlx's quantizer writes twice; their odd byte
    lives in the header and lane 1 of each reads it from there."""
    weight, scales = quantized_paired((2 * INNER, HIDDEN), seed=1)
    scales[0, 1] = mx.array(0x40, dtype=mx.uint8)
    scales[INNER, 1] = mx.array(0x41, dtype=mx.uint8)
    halved = halve_gate_up_scales(scales)
    assert halved is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    fused = nvfp4_halved_gate_up(x, weight, halved)

    assert_matches(fused, swiglu_reference(x, weight, scales))


def test_halving_refuses_an_unpaired_plane() -> None:
    _, scales = quantized_paired((2 * INNER, HIDDEN), seed=2)
    scales[7, 3] = mx.array(0x42, dtype=mx.uint8)
    assert halve_gate_up_scales(scales) is None


@pytest.mark.parametrize("code", [0x00, 0x01, 0x07, 0x0F, 0x38, 0x7E])
def test_scale_codes_match_dequantize(code: int) -> None:
    """`mx.quantize` never emits a denormal or a saturated scale, so the e4m3 decode's
    edges — including the `bits < 16` fast path — are only reachable by writing the scale
    plane directly. Codes with the sign bit set are outside the decode's domain and the
    halving refuses them."""
    weight, _ = quantized_paired((2 * INNER, HIDDEN), seed=3)
    scales = mx.full((2 * INNER, HIDDEN // GROUP), code, dtype=mx.uint8)
    halved = halve_gate_up_scales(scales)
    assert halved is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    fused = nvfp4_halved_gate_up(x, weight, halved)

    assert_matches(fused, swiglu_reference(x, weight, scales))
