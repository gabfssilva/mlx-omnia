"""Parity for the packed down/route/residual strategy against `gather_qmm` compositions.

The kernel is one dispatch for what is otherwise a routed `gather_qmm`, an unrouted
`quantized_matmul`, a weighted sum over experts and an add, so the reference is exactly
that composition — with the bf16 epilogue reproduced in the kernel's order, since the
routed total accumulates slot by slot and each product rounds before it lands.

The primitive carries no routed scaling of its own: like the affine sibling's spare slot,
the shared expert rides the last row of `act` and the model's scaling factor is already
inside the routing weights.
"""

import mlx.core as mx
import mlx.nn as nn
from conftest import relative_diff

from mlx_omnia.engine.core.kernels.down_combine import DownCombine, Nvfp4PackedDownCombine
from mlx_omnia.engine.core.kernels.down_combine.nvfp4_packed import applies, halve_down_scales
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear

HIDDEN = 64
INNER = 512
EXPERTS = 8
TOPK = 4
GROUP = 16
BITS = 4
SCALING = 2.5
SLACK = 2.0**-6


def paired_leaf(seed: int) -> QuantizedSwitchLinear:
    """An NVFP4 routed stack whose adjacent group pairs share a scale byte — what mlx's
    Metal `fp_quantize` produces, and the only shape the halved plane may carry."""
    mx.random.seed(seed)
    dense = SwitchLinear(EXPERTS, INNER, HIDDEN)
    dense.weight = mx.random.normal((EXPERTS, HIDDEN, INNER))
    leaf = dense.to_quantized(group_size=GROUP, bits=BITS, mode="nvfp4")
    leaf.scales = mx.repeat(leaf.scales[..., ::2], 2, axis=-1)
    mx.eval(leaf.weight, leaf.scales)
    return leaf


def paired_shared(seed: int) -> nn.QuantizedLinear:
    mx.random.seed(seed)
    dense = nn.Linear(INNER, HIDDEN, bias=False)
    dense.weight = mx.random.normal((HIDDEN, INNER))
    shared = nn.QuantizedLinear.from_linear(dense, group_size=GROUP, bits=BITS, mode="nvfp4")
    shared.scales = mx.repeat(shared.scales[..., ::2], 2, axis=-1)
    mx.eval(shared.weight, shared.scales)
    return shared


def build(
    leaf: QuantizedSwitchLinear, shared: nn.QuantizedLinear
) -> Nvfp4PackedDownCombine | None:
    return Nvfp4PackedDownCombine.build(
        leaf, hidden=HIDDEN, inner=INNER, bias=None, shared=shared, layout="interleaved"
    )


def assert_matches(ours: mx.array, expected: mx.array) -> None:
    """A scale plane of zeros makes both sides identically zero, and the house metric
    divides by the reference's magnitude — there is no relative bound to take there."""
    if float(mx.max(mx.abs(expected))) == 0.0:
        assert float(mx.max(mx.abs(ours))) == 0.0
        return
    assert relative_diff(ours, expected) < SLACK


def down_reference(
    act: mx.array, leaf: QuantizedSwitchLinear, indices: mx.array
) -> mx.array:
    return mx.gather_qmm(
        act.astype(mx.float32)[None, :, None],
        leaf.weight,
        leaf.scales,
        None,
        rhs_indices=indices[None],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    ).reshape(indices.size, HIDDEN)


def shared_reference(act: mx.array, shared: nn.QuantizedLinear) -> mx.array:
    return mx.gather_qmm(
        act.astype(mx.float32)[None, None, None],
        shared.weight[None],
        shared.scales[None],
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
    return residual + (total + shared)


def inputs(seed: int, experts: list[int]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """The declaration's spare row: `act[-1]` is the shared stack's activation and
    `chosen[-1]`/`weights[-1]` are the slot the packed kernel does not read."""
    mx.random.seed(seed)
    act = mx.random.normal((TOPK + 1, INNER)).astype(mx.bfloat16)
    chosen = mx.array([*experts, 0], dtype=mx.uint32)
    routing = (mx.random.uniform(shape=(TOPK,)) * SCALING).astype(mx.float32)
    weights = mx.concatenate([routing, mx.ones((1,), dtype=mx.float32)])
    residual = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    return act, chosen, weights, residual


def test_applies_pins_the_unlooped_contraction() -> None:
    assert applies(HIDDEN, INNER, TOPK)
    assert not applies(HIDDEN, INNER // 2, TOPK)
    assert not applies(HIDDEN + 1, INNER, TOPK)
    assert not applies(HIDDEN, INNER, 32)


def test_the_facade_resolves_it_only_with_a_shared_leaf() -> None:
    """The unrouted slot always runs, so the shared expert is part of the primitive here,
    not an optimization the kernel may skip."""
    leaf = paired_leaf(seed=0)
    shared = paired_shared(seed=1)
    with_shared = DownCombine(leaf, hidden=HIDDEN, inner=INNER, shared=shared)
    without = DownCombine(leaf, hidden=HIDDEN, inner=INNER)
    assert isinstance(with_shared.strategy, Nvfp4PackedDownCombine)
    assert not isinstance(without.strategy, Nvfp4PackedDownCombine)


def test_matches_gather_qmm_composition() -> None:
    leaf = paired_leaf(seed=0)
    shared = paired_shared(seed=1)
    strategy = build(leaf, shared)
    assert strategy is not None
    act, chosen, weights, residual = inputs(seed=2, experts=[0, 2, 5, 7])

    fused = strategy(act, chosen, weights, residual)

    routed = down_reference(act[:TOPK], leaf, chosen[:TOPK]).astype(mx.bfloat16)
    unrouted = shared_reference(act[TOPK], shared).astype(mx.bfloat16)
    assert_matches(fused, combine_reference(routed, unrouted, weights[:TOPK], residual))


def test_patch_header_restores_the_first_span() -> None:
    """Row 0 of the first expert is the one span mlx's quantizer writes twice; its odd byte
    lives in the header, for the routed plane and the unrouted one alike."""
    leaf = paired_leaf(seed=3)
    shared = paired_shared(seed=4)
    leaf.scales[0, 0, 1] = mx.array(0x40, dtype=mx.uint8)
    shared.scales[0, 1] = mx.array(0x41, dtype=mx.uint8)
    strategy = build(leaf, shared)
    assert strategy is not None
    # Expert 0 in a slot, so the patched routed row is actually read.
    act, chosen, weights, residual = inputs(seed=5, experts=[3, 0, 5, 7])

    fused = strategy(act, chosen, weights, residual)

    routed = down_reference(act[:TOPK], leaf, chosen[:TOPK]).astype(mx.bfloat16)
    unrouted = shared_reference(act[TOPK], shared).astype(mx.bfloat16)
    assert_matches(fused, combine_reference(routed, unrouted, weights[:TOPK], residual))


def test_build_refuses_an_unpaired_plane() -> None:
    leaf = paired_leaf(seed=6)
    shared = paired_shared(seed=7)
    leaf.scales[1, 3, 5] = mx.array(0x42, dtype=mx.uint8)
    assert halve_down_scales(leaf.scales) is None
    assert build(leaf, shared) is None
