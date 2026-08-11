"""The `GateUp` facade: the resolution table (which declaration gets which strategy),
the default's bit-exact agreement with the primitive written as explicit ops (which
pins the epilogue order — bias, clamp-before-activation, silu/swiglu_oai), and each
specialized kernel's fp32 parity against that same default. The delegator is total:
every declaration resolves, the default serving whatever no kernel does.
"""

import mlx.core as mx
from conftest import relative_diff

from mlx_omnia.core.kernels.gate_up import (
    Activation,
    AffineGateUp,
    DefaultGateUp,
    GateUp,
    Mxfp4GateUp,
    Nvfp4GateUp,
)
from mlx_omnia.core.layers import QuantizedSwitchLinear, SwitchLinear

EXPERTS = 8
ACTIVE = 4
HIDDEN = 512
INNER = 64


def leaf(mode: str, group_size: int, bits: int) -> QuantizedSwitchLinear:
    mx.random.seed(7)
    dense = SwitchLinear(EXPERTS, HIDDEN, 2 * INNER)
    quantized = dense.to_quantized(group_size=group_size, bits=bits, mode=mode)
    mx.eval(quantized.parameters())
    return quantized


def row_and_chosen() -> tuple[mx.array, mx.array]:
    mx.random.seed(11)
    row = mx.random.normal((HIDDEN,)).astype(mx.float32)
    chosen = mx.array([1, 3, 4, 6], dtype=mx.uint32)
    mx.eval(row, chosen)
    return row, chosen


def default(
    quantized: SwitchLinear | QuantizedSwitchLinear,
    activation: Activation = "silu",
    limit: float | None = None,
    bias: mx.array | None = None,
) -> DefaultGateUp:
    return DefaultGateUp.build(
        quantized,
        hidden=HIDDEN,
        inner=INNER,
        activation=activation,
        limit=limit,
        bias=bias,
        layout="interleaved",
        routing=None,
        shared=None,
    )


def test_affine_resolves_and_matches_default() -> None:
    quantized = leaf("affine", group_size=64, bits=4)
    gate_up = GateUp(quantized, hidden=HIDDEN, inner=INNER)
    assert isinstance(gate_up.strategy, AffineGateUp)
    row, chosen = row_and_chosen()
    assert relative_diff(gate_up(row, chosen), default(quantized)(row, chosen)) < 1e-5


def test_affine_clamped_resolves_and_matches_default() -> None:
    quantized = leaf("affine", group_size=64, bits=4)
    limit = 0.02  # small enough to bite at these magnitudes
    gate_up = GateUp(quantized, hidden=HIDDEN, inner=INNER, limit=limit)
    assert isinstance(gate_up.strategy, AffineGateUp)
    row, chosen = row_and_chosen()
    clamped = default(quantized, limit=limit)(row, chosen)
    assert relative_diff(clamped, default(quantized)(row, chosen)) > 1e-3  # the limit bites
    assert relative_diff(gate_up(row, chosen), clamped) < 1e-5


def test_nvfp4_resolves_and_matches_default() -> None:
    quantized = leaf("nvfp4", group_size=16, bits=4)
    gate_up = GateUp(quantized, hidden=HIDDEN, inner=INNER)
    assert isinstance(gate_up.strategy, Nvfp4GateUp)
    row, chosen = row_and_chosen()
    assert relative_diff(gate_up(row, chosen), default(quantized)(row, chosen)) < 1e-5


def test_mxfp4_swiglu_oai_resolves_and_matches_default() -> None:
    quantized = leaf("mxfp4", group_size=32, bits=4)
    mx.random.seed(13)
    bias = mx.random.normal((EXPERTS, 2 * INNER)).astype(mx.float32)
    mx.eval(bias)
    gate_up = GateUp(
        quantized, hidden=HIDDEN, inner=INNER,
        activation="swiglu_oai", limit=7.0, bias=bias,
    )
    assert isinstance(gate_up.strategy, Mxfp4GateUp)
    row, chosen = row_and_chosen()
    reference = default(quantized, activation="swiglu_oai", limit=7.0, bias=bias)
    assert relative_diff(gate_up(row, chosen), reference(row, chosen)) < 1e-5


def test_mxfp4_silu_falls_to_default() -> None:
    # No silu mxfp4 kernel exists yet: the sibling bakes swiglu_oai.
    quantized = leaf("mxfp4", group_size=32, bits=4)
    assert isinstance(GateUp(quantized, hidden=HIDDEN, inner=INNER).strategy, DefaultGateUp)


def test_swiglu_oai_on_affine_falls_to_default() -> None:
    quantized = leaf("affine", group_size=64, bits=4)
    mx.random.seed(13)
    bias = mx.random.normal((EXPERTS, 2 * INNER)).astype(mx.float32)
    facade = GateUp(
        quantized, hidden=HIDDEN, inner=INNER,
        activation="swiglu_oai", limit=7.0, bias=bias,
    )
    assert isinstance(facade.strategy, DefaultGateUp)


def test_nvfp4_with_clamp_falls_to_default() -> None:
    quantized = leaf("nvfp4", group_size=16, bits=4)
    facade = GateUp(quantized, hidden=HIDDEN, inner=INNER, limit=7.0)
    assert isinstance(facade.strategy, DefaultGateUp)


def test_dense_leaf_falls_to_default() -> None:
    dense = SwitchLinear(EXPERTS, HIDDEN, 2 * INNER)
    assert isinstance(GateUp(dense, hidden=HIDDEN, inner=INNER).strategy, DefaultGateUp)


def test_untiled_shape_falls_to_default() -> None:
    # hidden 384 breaks the 512-multiple tiling the affine gate-up reduction needs.
    mx.random.seed(7)
    narrow = SwitchLinear(EXPERTS, 384, 2 * INNER).to_quantized(
        group_size=64, bits=4, mode="affine"
    )
    assert isinstance(GateUp(narrow, hidden=384, inner=INNER).strategy, DefaultGateUp)


def test_default_silu_is_the_ops_chain_bit_for_bit() -> None:
    """The default runs the leaf's own gather and the declared epilogue in ops; a
    reordered epilogue (clamp after the activation, bias after the split) breaks
    exact equality here."""
    quantized = leaf("affine", group_size=64, bits=4)
    row, chosen = row_and_chosen()
    limit = 0.02
    assert quantized.biases is not None
    fused = mx.gather_qmm(
        row[None, None, None], quantized.weight, quantized.scales, quantized.biases,
        rhs_indices=chosen[None], transpose=True, group_size=64, bits=4,
    ).reshape(ACTIVE, INNER, 2)
    gate = mx.minimum(fused[..., 0], limit)
    up = mx.clip(fused[..., 1], -limit, limit)
    reference = gate * mx.sigmoid(gate) * up
    unclamped = fused[..., 0] * mx.sigmoid(fused[..., 0]) * fused[..., 1]
    assert relative_diff(reference, unclamped) > 1e-3
    ours = default(quantized, limit=limit)(row, chosen)
    assert mx.array_equal(ours, reference)


def test_default_swiglu_oai_is_the_ops_chain_bit_for_bit() -> None:
    quantized = leaf("affine", group_size=64, bits=4)
    row, chosen = row_and_chosen()
    mx.random.seed(13)
    bias = mx.random.normal((EXPERTS, 2 * INNER)).astype(mx.float32)
    limit = 0.02
    assert quantized.biases is not None
    fused = mx.gather_qmm(
        row[None, None, None], quantized.weight, quantized.scales, quantized.biases,
        rhs_indices=chosen[None], transpose=True, group_size=64, bits=4,
    ).reshape(ACTIVE, 2 * INNER) + bias[chosen]
    pairs = fused.reshape(ACTIVE, INNER, 2)
    gate = mx.minimum(pairs[..., 0], limit)
    up = mx.clip(pairs[..., 1], -limit, limit)
    reference = gate * mx.sigmoid(1.702 * gate) * (up + 1)
    ours = default(quantized, activation="swiglu_oai", limit=limit, bias=bias)(row, chosen)
    assert mx.array_equal(ours, reference)
