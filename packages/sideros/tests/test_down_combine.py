"""The `DownCombine` facade: the resolution table (which declaration gets which
strategy), the default's bit-exact agreement with the primitive written as explicit
ops (which pins the epilogue order — bias before the routing weight, residual last),
and each specialized kernel's fp32 parity against that same default. The delegator is
total: every declaration resolves, the default serving whatever no kernel does —
including a shared expert quantized unlike the routed stack, which the affine spare
slot refuses.
"""

import mlx.core as mx
import mlx.nn as nn
from conftest import relative_diff

from sideros.core.kernels.down_combine import (
    AffineDownCombine,
    DefaultDownCombine,
    DownCombine,
    Mxfp4DownCombine,
    Nvfp4DownCombine,
)
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear

EXPERTS = 8
ACTIVE = 4
HIDDEN = 64
INNER = 256


def leaf(mode: str, group_size: int, bits: int) -> QuantizedSwitchLinear:
    mx.random.seed(7)
    dense = SwitchLinear(EXPERTS, INNER, HIDDEN)
    quantized = dense.to_quantized(group_size=group_size, bits=bits, mode=mode)
    mx.eval(quantized.parameters())
    return quantized


def shared_down(group_size: int, bits: int) -> nn.QuantizedLinear:
    mx.random.seed(17)
    linear = nn.QuantizedLinear(INNER, HIDDEN, bias=False, group_size=group_size, bits=bits)
    mx.eval(linear.parameters())
    return linear


def step_inputs(rows: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(11)
    act = mx.random.normal((rows, INNER)).astype(mx.float32)
    chosen = mx.array([1, 3, 4, 6, EXPERTS][:rows], dtype=mx.uint32)
    weights = mx.random.uniform(shape=(rows,)).astype(mx.float32)
    residual = mx.random.normal((HIDDEN,)).astype(mx.float32)
    mx.eval(act, chosen, weights, residual)
    return act, chosen, weights, residual


def default(
    quantized: SwitchLinear | QuantizedSwitchLinear,
    bias: mx.array | None = None,
    shared: nn.Linear | nn.QuantizedLinear | None = None,
) -> DefaultDownCombine:
    return DefaultDownCombine.build(
        quantized, hidden=HIDDEN, inner=INNER, bias=bias, shared=shared
    )


def test_affine_resolves_and_matches_default() -> None:
    quantized = leaf("affine", group_size=64, bits=4)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER)
    assert isinstance(down.strategy, AffineDownCombine)
    act, chosen, weights, residual = step_inputs(ACTIVE)
    reference = default(quantized)(act, chosen, weights, residual)
    assert relative_diff(down(act, chosen, weights, residual), reference) < 1e-5


def test_affine_shared_slot_resolves_and_matches_default() -> None:
    """The spare slot against the default's independent path: the same shared expert
    computed as its own layer call must land within the fp32 bound."""
    quantized = leaf("affine", group_size=64, bits=4)
    shared = shared_down(group_size=64, bits=4)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER, shared=shared)
    assert isinstance(down.strategy, AffineDownCombine)
    act, chosen, weights, residual = step_inputs(ACTIVE + 1)
    reference = default(quantized, shared=shared)(act, chosen, weights, residual)
    assert relative_diff(down(act, chosen, weights, residual), reference) < 1e-5


def test_nvfp4_resolves_and_matches_default() -> None:
    quantized = leaf("nvfp4", group_size=16, bits=4)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER)
    assert isinstance(down.strategy, Nvfp4DownCombine)
    act, chosen, weights, residual = step_inputs(ACTIVE)
    reference = default(quantized)(act, chosen, weights, residual)
    assert relative_diff(down(act, chosen, weights, residual), reference) < 1e-5


def test_mxfp4_resolves_and_matches_default() -> None:
    quantized = leaf("mxfp4", group_size=32, bits=4)
    mx.random.seed(13)
    bias = mx.random.normal((EXPERTS, HIDDEN)).astype(mx.float32)
    mx.eval(bias)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER, bias=bias)
    assert isinstance(down.strategy, Mxfp4DownCombine)
    act, chosen, weights, residual = step_inputs(ACTIVE)
    reference = default(quantized, bias=bias)(act, chosen, weights, residual)
    assert relative_diff(down(act, chosen, weights, residual), reference) < 1e-5


def test_shared_with_mismatched_format_falls_to_default() -> None:
    quantized = leaf("affine", group_size=64, bits=4)
    mismatched = shared_down(group_size=64, bits=8)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER, shared=mismatched)
    assert isinstance(down.strategy, DefaultDownCombine)


def test_nvfp4_with_shared_falls_to_default() -> None:
    quantized = leaf("nvfp4", group_size=16, bits=4)
    shared = shared_down(group_size=64, bits=4)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER, shared=shared)
    assert isinstance(down.strategy, DefaultDownCombine)


def test_mxfp4_without_bias_falls_to_default() -> None:
    quantized = leaf("mxfp4", group_size=32, bits=4)
    down = DownCombine(quantized, hidden=HIDDEN, inner=INNER)
    assert isinstance(down.strategy, DefaultDownCombine)


def test_dense_leaf_falls_to_default() -> None:
    dense = SwitchLinear(EXPERTS, INNER, HIDDEN)
    assert isinstance(DownCombine(dense, hidden=HIDDEN, inner=INNER).strategy, DefaultDownCombine)


def test_untiled_shape_falls_to_default() -> None:
    # inner 192 breaks the 256-multiple tiling the affine down reduction needs.
    mx.random.seed(7)
    narrow = SwitchLinear(EXPERTS, 192, HIDDEN).to_quantized(
        group_size=64, bits=4, mode="affine"
    )
    assert isinstance(DownCombine(narrow, hidden=HIDDEN, inner=192).strategy, DefaultDownCombine)


def test_default_is_the_ops_chain_bit_for_bit() -> None:
    """The default runs the leaf's own gather and the combine in ops; a reordered
    epilogue (bias after the routing weight, a dropped shared row) breaks exact
    equality here."""
    quantized = leaf("affine", group_size=64, bits=4)
    shared = shared_down(group_size=64, bits=8)
    mx.random.seed(13)
    bias = mx.random.normal((EXPERTS, HIDDEN)).astype(mx.float32)
    act, chosen, weights, residual = step_inputs(ACTIVE + 1)
    assert quantized.biases is not None
    projected = mx.gather_qmm(
        act[None, :ACTIVE, None], quantized.weight, quantized.scales, quantized.biases,
        rhs_indices=chosen[None, :ACTIVE], transpose=True, group_size=64, bits=4,
    ).reshape(ACTIVE, HIDDEN) + bias[chosen[:ACTIVE]]
    reference = (projected * weights[:ACTIVE, None]).sum(axis=0)
    reference = reference + weights[-1] * shared(act[ACTIVE][None]).reshape(-1)
    reference = reference + residual
    ours = default(quantized, bias=bias, shared=shared)(act, chosen, weights, residual)
    assert mx.array_equal(ours, reference)
