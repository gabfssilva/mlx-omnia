"""The one-row affine INT8 projection against `mx.quantized_matmul`'s own arithmetic.

Both paths read the same packed tensors, so what is compared is arithmetic order, not the
quantizer. The kernel accumulates each 256-value K block in the same sequence the stock
`qmv_fast` does and seeds the accumulator with the first product instead of adding to a
dead zero, which differs only in the sign of a zero — the fp32 template is expected to
match exactly, not within a tolerance.
"""

import mlx.core as mx
import pytest
from conftest import relative_diff

from sideros.core.kernels.int8_qmv import (
    gated_int8_qmv,
    gated_int8_qmv_applies,
    int8_qmv,
    int8_qmv_applies,
    norm_int8_qmv,
)
from sideros.core.patch import original

GROUP = 32
BITS = 8


def packed(rows: int, kdim: int, seed: int) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    dense = mx.random.normal((rows, kdim)).astype(mx.float32)
    weight, scales, biases = mx.quantize(dense, group_size=GROUP, bits=BITS, mode="affine")
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def reference(
    x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array
) -> mx.array:
    return original(mx, "quantized_matmul")(
        x, weight, scales=scales, biases=biases, transpose=True, group_size=GROUP, bits=BITS
    )


def test_applies_states_its_contract() -> None:
    assert int8_qmv_applies(2048, 512, group_size=GROUP, bits=BITS, rows_per_simd=4)
    assert not int8_qmv_applies(2048, 512, group_size=64, bits=BITS, rows_per_simd=4)
    assert not int8_qmv_applies(2048, 512, group_size=GROUP, bits=4, rows_per_simd=4)
    # kdim must cover whole 256-value blocks, rows the two simdgroups' tiling.
    assert not int8_qmv_applies(2048 + 128, 512, group_size=GROUP, bits=BITS, rows_per_simd=4)
    assert not int8_qmv_applies(2048, 12, group_size=GROUP, bits=BITS, rows_per_simd=4)


@pytest.mark.parametrize("rows_per_simd", [4, 1])
@pytest.mark.parametrize("kdim", [256, 512, 2048])
@pytest.mark.parametrize("rows", [8, 512])
def test_matches_quantized_matmul_fp32(rows: int, kdim: int, rows_per_simd: int) -> None:
    weight, scales, biases = packed(rows, kdim, seed=kdim + rows)
    x = mx.random.normal((1, kdim)).astype(mx.float32)
    ours = int8_qmv(x, weight, scales, biases, rows_per_simd=rows_per_simd)
    assert relative_diff(ours, reference(x, weight, scales, biases)) < 1e-6


@pytest.mark.parametrize("shape", [(2048,), (1, 2048), (1, 1, 2048)])
def test_output_shape_follows_the_input(shape: tuple[int, ...]) -> None:
    weight, scales, biases = packed(512, 2048, seed=7)
    x = mx.random.normal(shape).astype(mx.float32)
    ours = int8_qmv(x, weight, scales, biases)
    assert ours.shape == (*shape[:-1], 512)
    assert relative_diff(ours, reference(x, weight, scales, biases)) < 1e-6


def test_matches_in_bfloat16() -> None:
    """bf16 in, bf16 out: the accumulation is fp32 on both sides, so the only gap is the
    final cast, and the two paths round the same sum the same way."""
    weight, scales, biases = packed(512, 2048, seed=11)
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 2048)).astype(mx.bfloat16)
    ours = int8_qmv(x, weight, scales, biases)
    assert ours.dtype == mx.bfloat16
    assert relative_diff(ours, reference(x, weight, scales, biases)) < 1e-2


def test_gated_projection_matches_two_dispatch_chain() -> None:
    heads = 4
    head_dim = 128
    rows = 512
    weight, scales, biases = packed(rows, heads * head_dim, seed=29)
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    attention = mx.random.normal((1, 1, heads * head_dim)).astype(mx.bfloat16)
    gate_logits = mx.random.normal((1, 1, heads)).astype(mx.bfloat16)
    gate = mx.logaddexp(gate_logits.astype(mx.float32), mx.array(0.0)).astype(mx.bfloat16)
    gated = (attention.reshape(heads, head_dim) * gate.reshape(heads, 1)).reshape(
        attention.shape
    )

    ours = gated_int8_qmv(attention, gate_logits, weight, scales, biases)
    preactivated = gated_int8_qmv(
        attention, gate, weight, scales, biases, preactivated=True
    )

    assert gated_int8_qmv_applies(attention, gate_logits, weight, scales, biases)
    assert ours.shape == (1, 1, rows)
    assert relative_diff(ours, reference(gated, weight, scales, biases)) < 2.0**-7
    assert relative_diff(preactivated, reference(gated, weight, scales, biases)) < 2.0**-7


def test_norm_projection_matches_two_dispatch_chain() -> None:
    kdim = 2048
    rows = 512
    weight, scales, biases = packed(rows, kdim, seed=31)
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 1, kdim)).astype(mx.bfloat16)
    norm_weight = mx.random.normal((kdim,)).astype(mx.bfloat16)
    normalized = mx.fast.rms_norm(x, norm_weight, 1.0e-6)

    ours = norm_int8_qmv(x, norm_weight, weight, scales, biases)

    assert ours.shape == (1, 1, rows)
    assert relative_diff(ours, reference(normalized, weight, scales, biases)) < 2.0**-7
