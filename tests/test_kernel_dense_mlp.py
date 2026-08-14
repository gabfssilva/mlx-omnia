"""The fused dense bf16 MLP against the op chain it replaces.

Both kernels accumulate in fp32 and round to bf16 exactly where the op chain does, so the
reference has to round at the same points -- an fp32 SwiGLU compared against a bf16 one
measures the reference, not the kernel. `bf16_step` below is that emulation: every
intermediate of `silu = gate / (1 + exp(-gate))` is rounded, because in the kernel every
one of them is a `bfloat`.

The floor is one bf16 ulp relative (2^-8) doubled: what remains after matching the rounding
points is the fp32 accumulation order (mlx's gemv walks the contraction differently from
this tiling) and `metal::exp` against `mx.exp`, both of which survive the bf16 rounding
only when they straddle a tie.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.mlp import DefaultMlp, Mlp
from mlx_omnia.engine.core.kernels.mlp.dense import (
    DenseMlp,
    dense_down_residual,
    dense_down_residual_applies,
    dense_gate_up_swiglu,
    dense_gate_up_swiglu_applies,
)
from mlx_omnia.engine.core.layers import SwiGLU
from tests.conftest import relative_diff

HIDDEN = 256
INNER = 512
FLOOR = 2.0**-7


def bf16_step(value: mx.array) -> mx.array:
    """One bf16 rounding, back in fp32 so the next step's arithmetic is exact."""
    return value.astype(mx.bfloat16).astype(mx.float32)


def swiglu_reference(gate: mx.array, up: mx.array) -> mx.array:
    """The kernel's epilogue, step for step: gate and up rounded, then the stable
    `exp(|g|)` branch form with a bf16 rounding after every operation."""
    gate, up = bf16_step(gate), bf16_step(up)
    exp_abs = bf16_step(mx.exp(mx.abs(gate)))
    denominator = bf16_step(1.0 + exp_abs)
    y = bf16_step(1.0 / denominator)
    sigmoid = mx.where(gate < 0.0, y, bf16_step(1.0 - y))
    silu = bf16_step(gate * sigmoid)
    return bf16_step(silu * up)


def test_applies_requires_whole_tiles_and_bf16() -> None:
    assert dense_gate_up_swiglu_applies(HIDDEN, INNER, mx.bfloat16)
    assert not dense_gate_up_swiglu_applies(HIDDEN, INNER, mx.float32)
    assert not dense_gate_up_swiglu_applies(HIDDEN + 4, INNER, mx.bfloat16)
    assert not dense_gate_up_swiglu_applies(HIDDEN, INNER + 4, mx.bfloat16)

    assert dense_down_residual_applies(HIDDEN, INNER, mx.bfloat16)
    assert not dense_down_residual_applies(HIDDEN, INNER, mx.float32)
    assert not dense_down_residual_applies(HIDDEN + 4, INNER, mx.bfloat16)
    assert not dense_down_residual_applies(HIDDEN, INNER + 4, mx.bfloat16)


def test_gate_up_swiglu_matches_the_op_chain() -> None:
    mx.random.seed(0)
    fused = (mx.random.normal((2 * INNER, HIDDEN)) * 0.1).astype(mx.bfloat16)
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    activated = dense_gate_up_swiglu(x, fused)

    projected = mx.matmul(fused.astype(mx.float32), x.astype(mx.float32))
    expected = swiglu_reference(projected[:INNER], projected[INNER:])
    assert relative_diff(activated, expected) < FLOOR


def test_down_residual_matches_the_op_chain() -> None:
    mx.random.seed(1)
    weight = (mx.random.normal((HIDDEN, INNER)) * 0.1).astype(mx.bfloat16)
    activated = mx.random.normal((INNER,)).astype(mx.bfloat16)
    residual = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    out = dense_down_residual(activated, weight, residual)

    projected = mx.matmul(weight.astype(mx.float32), activated.astype(mx.float32))
    expected = bf16_step(bf16_step(projected) + residual.astype(mx.float32))
    assert relative_diff(out, expected) < FLOOR


def test_gate_up_layout_is_gate_rows_first() -> None:
    """Row layout, not arithmetic. SwiGLU is asymmetric in its two operands, so a kernel
    that read the stack the other way round would still match a symmetric reference; the
    swapped reference has to disagree."""
    mx.random.seed(2)
    gate = (mx.random.normal((INNER, HIDDEN)) * 0.1).astype(mx.bfloat16)
    up = (mx.random.normal((INNER, HIDDEN)) * 0.1).astype(mx.bfloat16)
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    activated = dense_gate_up_swiglu(x, mx.concatenate([gate, up], axis=0))

    g = mx.matmul(gate.astype(mx.float32), x.astype(mx.float32))
    u = mx.matmul(up.astype(mx.float32), x.astype(mx.float32))
    assert relative_diff(activated, swiglu_reference(g, u)) < FLOOR
    assert relative_diff(activated, swiglu_reference(u, g)) > FLOOR


def test_facade_resolves_the_dense_kernels_and_matches_the_leaf() -> None:
    """The delegator is total: a bf16 leaf whose shapes tile gets the kernels, anything
    else falls through to the leaf's own call, and both compute the same MLP step."""
    mx.random.seed(3)
    leaf = SwiGLU(HIDDEN, INNER)
    leaf.set_dtype(mx.bfloat16)
    row = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    residual = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)

    fused = Mlp(leaf, hidden=HIDDEN, inner=INNER)
    assert isinstance(fused.strategy, DenseMlp)

    plain = Mlp(leaf, hidden=HIDDEN, inner=INNER, activation="gelu_tanh")
    assert isinstance(plain.strategy, DefaultMlp)

    # Against the leaf, not against a rounding-matched reference: the two chains round
    # at different points, so what is claimed here is the same MLP, not the same bits.
    assert relative_diff(fused(row, residual), plain(row, residual)) < 2.0**-5


def test_facade_falls_back_when_the_leaf_is_not_bf16() -> None:
    leaf = SwiGLU(HIDDEN, INNER)
    assert isinstance(Mlp(leaf, hidden=HIDDEN, inner=INNER).strategy, DefaultMlp)
