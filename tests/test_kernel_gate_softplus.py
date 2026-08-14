"""The fused INT8 projection + softplus against the two-dispatch chain it replaces.

`mx.quantized_matmul(..., transpose=True, group_size=32, bits=8)` then `nn.softplus` over
the same packed tensors, so what is compared is the fusion and the accumulation order.
The kernel rounds its fp32 accumulator to bf16 *before* the activation, which is where the
two-dispatch chain's projection output rounds too; comparing against a chain that stayed
in fp32 through the activation would measure a different function.

The floor is the bf16 epilogue: both sides round once to bf16, `relative_diff` normalizes
by the row maximum, and softplus is 1-Lipschitz, so a one-ulp disagreement on the largest
element is `2^-8` and two of them bound anything the fp32 orders can produce.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.core.kernels.qmv.softplus import gate_softplus, gate_softplus_applies
from mlx_omnia.engine.core.patch import original
from tests.conftest import relative_diff

KDIM = 2048
ROWS = 64
GROUP = 32
BITS = 8
BF16_FLOOR = 2.0**-7


def packed(rows: int, kdim: int, seed: int) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    dense = mx.random.normal((rows, kdim)).astype(mx.float32)
    weight, scales, biases = mx.quantize(dense, group_size=GROUP, bits=BITS, mode="affine")
    mx.eval(weight, scales, biases)
    return weight, scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)


def logits(
    x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array
) -> mx.array:
    """`original`, not `mx.quantized_matmul`: importing mlx_omnia installs a replacement
    over that name, and comparing against it would compare a kernel with itself."""
    projected = original(mx, "quantized_matmul")(
        x, weight, scales=scales, biases=biases, transpose=True, group_size=GROUP, bits=BITS
    )
    assert isinstance(projected, mx.array)
    return projected


def reference(
    x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array
) -> mx.array:
    activation = logits(x, weight, scales, biases).astype(mx.float32)
    return nn.softplus(activation).astype(mx.bfloat16)


def test_applies_states_its_contract() -> None:
    assert gate_softplus_applies(KDIM, ROWS, group_size=GROUP, bits=BITS)
    assert not gate_softplus_applies(KDIM, ROWS, group_size=GROUP, bits=4)
    # A lane's eight values have to sit in one group, and a block in whole groups.
    assert not gate_softplus_applies(KDIM, ROWS, group_size=12, bits=BITS)
    assert not gate_softplus_applies(KDIM, ROWS, group_size=512, bits=BITS)
    # 256 values per simdgroup step, eight output rows per threadgroup.
    assert not gate_softplus_applies(KDIM + 128, ROWS, group_size=GROUP, bits=BITS)
    assert not gate_softplus_applies(KDIM, ROWS + 4, group_size=GROUP, bits=BITS)


@pytest.mark.parametrize("kdim", [256, 512, KDIM])
@pytest.mark.parametrize("rows", [8, ROWS])
def test_matches_the_chain(rows: int, kdim: int) -> None:
    weight, scales, biases = packed(rows, kdim, seed=rows + kdim)
    x = mx.random.normal((1, 1, kdim)).astype(mx.bfloat16)

    ours = gate_softplus(x, weight, scales, biases)

    assert ours.shape == (1, 1, rows)
    assert ours.dtype == mx.bfloat16
    assert relative_diff(ours, reference(x, weight, scales, biases)) < BF16_FLOOR


def test_matches_the_chain_where_softplus_saturates() -> None:
    """Scaling the activation pushes the logits far from zero in both directions, where
    softplus is the identity on one side and flushes to zero on the other -- the two arms
    of the `hi + log1p(exp(lo - hi))` form."""
    weight, scales, biases = packed(ROWS, KDIM, seed=17)
    x = (mx.random.normal((1, 1, KDIM)) * 40.0).astype(mx.bfloat16)

    ours = gate_softplus(x, weight, scales, biases)

    assert relative_diff(ours, reference(x, weight, scales, biases)) < BF16_FLOOR


def test_output_shape_follows_the_input() -> None:
    weight, scales, biases = packed(ROWS, KDIM, seed=23)
    x = mx.random.normal((KDIM,)).astype(mx.bfloat16)

    assert gate_softplus(x, weight, scales, biases).shape == (ROWS,)


@pytest.mark.parametrize("batch", [2, 4])
def test_small_row_batch_matches_the_chain(batch: int) -> None:
    weight, scales, biases = packed(ROWS, KDIM, seed=29)
    x = mx.random.normal((batch, 1, KDIM)).astype(mx.bfloat16)

    ours = gate_softplus(x, weight, scales, biases)

    assert ours.shape == (batch, 1, ROWS)
    assert relative_diff(ours, reference(x, weight, scales, biases)) < BF16_FLOOR
