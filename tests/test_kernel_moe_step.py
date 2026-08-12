"""The T=1 sparse step's nvfp4 kernels against the op chain.

The projections claim mlx's own `fp_qmv_fast` tiling — same groups, same fp32
accumulation order — and the epilogues claim the op chain's rounding, so the comparison
is equality on the bf16 output, not a tolerance. The shared expert rides the same two
dispatches when declared, and its arithmetic is asserted the same way.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.core.kernels.moe_step import DefaultMoeStep, Nvfp4MoeStep
from mlx_omnia.engine.core.layers import SwitchLinear

EXPERTS = 8
HIDDEN = 128
INNER = 96
SHARED_INNER = 160
K = 3


def _quantized(out_dims: int, in_dims: int) -> nn.QuantizedLinear:
    return nn.QuantizedLinear.from_linear(
        nn.Linear(in_dims, out_dims, bias=False), group_size=16, bits=4, mode="nvfp4"
    )


@pytest.mark.parametrize("with_shared", [False, True])
def test_nvfp4_step_matches_ops(with_shared: bool) -> None:
    mx.random.seed(5)
    q1 = SwitchLinear(EXPERTS, HIDDEN, INNER).to_quantized(
        group_size=16, bits=4, mode="nvfp4"
    )
    q2 = SwitchLinear(EXPERTS, INNER, HIDDEN).to_quantized(
        group_size=16, bits=4, mode="nvfp4"
    )
    shared = (
        (_quantized(SHARED_INNER, HIDDEN), _quantized(HIDDEN, SHARED_INNER))
        if with_shared
        else None
    )
    fused = Nvfp4MoeStep.build(fc1=q1, fc2=q2, hidden=HIDDEN, inner=INNER, shared=shared)
    assert fused is not None
    reference = DefaultMoeStep.build(
        fc1=q1, fc2=q2, hidden=HIDDEN, inner=INNER, shared=shared
    )
    assert reference is not None

    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    chosen = mx.array([1, 4, 6], dtype=mx.uint32)
    weights = mx.array([0.5, 0.3, 0.2], dtype=mx.float32) * 2.5

    out = fused(x, chosen, weights)
    wanted = reference(x, chosen, weights)
    assert out.shape == wanted.shape
    assert mx.array_equal(out, wanted).item()


def test_rows_variant_matches_per_row() -> None:
    """T rows through the rows kernels equal the single-row path row for row — the
    equality the verification forward's fidelity to the decode rests on."""
    mx.random.seed(5)
    q1 = SwitchLinear(EXPERTS, HIDDEN, INNER).to_quantized(
        group_size=16, bits=4, mode="nvfp4"
    )
    q2 = SwitchLinear(EXPERTS, INNER, HIDDEN).to_quantized(
        group_size=16, bits=4, mode="nvfp4"
    )
    fused = Nvfp4MoeStep.build(fc1=q1, fc2=q2, hidden=HIDDEN, inner=INNER)
    assert fused is not None

    x = mx.random.normal((3, HIDDEN)).astype(mx.bfloat16)
    chosen = mx.array([[1, 4, 6], [0, 2, 5], [3, 4, 7]], dtype=mx.uint32)
    weights = (mx.random.uniform(shape=(3, K)) * 2.5).astype(mx.float32)

    batched = fused.rows(x, chosen, weights)
    for t in range(3):
        row = fused(x[t], chosen[t], weights[t])
        assert mx.array_equal(batched[t], row).item(), t
