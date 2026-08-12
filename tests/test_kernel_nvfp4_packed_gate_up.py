"""Parity for the packed routed gate/up strategy against `gather_qmm`'s own arithmetic.

Three things can only be wrong together with the reference: the bit-trick NVFP4 decode
(off by a power of two, never by a rounding), the walk-order scale bank (a scale byte read
one k-block or one sub away is a different group's), and the inline tournament (the wrong
expert entirely). So the comparison is against `mx.gather_qmm(..., mode="nvfp4")` over the
leaf's own tensors, with the expert set computed independently from the ordinal keys.

The leaf is row-interleaved as `SwitchGLU` stacks it; the tile bank the kernel walks is
built by `Nvfp4PackedGateUp.build`, so the relayout is under test too — a wrong permutation
reads another row's codes or another row's scales.

The kernel rounds its fp32 accumulator to bf16 once and then runs a bf16 SwiGLU chain; the
reference reproduces that chain expression for expression, so the only slack is the fp32
accumulation order upstream of the single rounding — one bf16 ulp of it, when a dot lands
near a rounding boundary.
"""

import random

import mlx.core as mx
from conftest import relative_diff

from mlx_omnia.engine.core.kernels.gate_up import GateUp, Nvfp4PackedGateUp, OrdinalRouting
from mlx_omnia.engine.core.kernels.gate_up.nvfp4_packed import applies
from mlx_omnia.engine.core.kernels.shared.nvfp4 import halved_group32_scales
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear

HIDDEN = 512
INNER = 32
EXPERTS = 64
TOPK = 4
GROUP = 16
BITS = 4
SLACK = 2.0**-6


def paired_leaf(seed: int) -> QuantizedSwitchLinear:
    """An NVFP4 leaf whose adjacent group pairs share a scale byte — what mlx's Metal
    `fp_quantize` produces, and the only shape the halved plane may carry."""
    mx.random.seed(seed)
    dense = SwitchLinear(EXPERTS, HIDDEN, 2 * INNER)
    dense.weight = mx.random.normal((EXPERTS, 2 * INNER, HIDDEN)).astype(mx.float32)
    leaf = dense.to_quantized(group_size=GROUP, bits=BITS, mode="nvfp4")
    leaf.scales = mx.repeat(leaf.scales[..., ::2], 2, axis=-1)
    mx.eval(leaf.weight, leaf.scales)
    return leaf


def build(leaf: QuantizedSwitchLinear) -> Nvfp4PackedGateUp | None:
    return Nvfp4PackedGateUp.build(
        leaf,
        hidden=HIDDEN,
        inner=INNER,
        activation="silu",
        limit=None,
        bias=None,
        layout="interleaved",
        routing=OrdinalRouting(TOPK),
        shared=None,
    )


def assert_matches(ours: mx.array, expected: mx.array) -> None:
    """A scale plane of zeros makes both sides identically zero, and the house metric
    divides by the reference's magnitude — there is no relative bound to take there."""
    if float(mx.max(mx.abs(expected))) == 0.0:
        assert float(mx.max(mx.abs(ours))) == 0.0
        return
    assert relative_diff(ours, expected) < SLACK


def swiglu_reference(
    x: mx.array, leaf: QuantizedSwitchLinear, experts: list[int]
) -> mx.array:
    """The leaf's own row interleave: gate on the even rows, up on the odd ones."""
    projected = mx.gather_qmm(
        x.astype(mx.float32)[None, None, None],
        leaf.weight,
        leaf.scales,
        None,
        rhs_indices=mx.array(experts, dtype=mx.uint32)[None],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
        mode="nvfp4",
    ).reshape(len(experts), 2 * INNER)
    gate = projected[:, 0::2].astype(mx.bfloat16)
    up = projected[:, 1::2].astype(mx.bfloat16)
    one = mx.array(1, dtype=mx.bfloat16)
    y = one / (one + mx.exp(mx.abs(gate)))
    sigmoid = mx.where(gate < 0, y, one - y)
    return (gate * sigmoid) * up


def selected(keys: list[int]) -> list[int]:
    """Ascending ordinal, ties to the lower expert index — the tournament's rule."""
    return sorted(range(len(keys)), key=lambda e: (keys[e], e))[:TOPK]


def shuffled_keys(seed: int) -> list[int]:
    """Distinct ordinals in a fixed shuffled order; `0` is below every one of them."""
    order = random.Random(seed).sample(range(EXPERTS), EXPERTS)
    return [value * 1000 + 5 for value in order]


def test_applies_requires_blocked_contraction() -> None:
    assert applies(HIDDEN, INNER, EXPERTS, TOPK)
    assert not applies(HIDDEN + 16, INNER, EXPERTS, TOPK)
    assert not applies(HIDDEN, INNER + 16, EXPERTS, TOPK)
    assert not applies(HIDDEN, INNER, EXPERTS + 16, TOPK)
    assert not applies(HIDDEN, INNER, EXPERTS, EXPERTS + 1)


def test_the_facade_resolves_it_only_on_the_routing_declaration() -> None:
    """Routing lives inside this kernel, so without keys to give it the leaf's format is
    not enough — the plain nvfp4 strategy takes the step instead."""
    leaf = paired_leaf(seed=0)
    declared = GateUp(
        leaf, hidden=HIDDEN, inner=INNER, routing=OrdinalRouting(TOPK)
    )
    plain = GateUp(leaf, hidden=HIDDEN, inner=INNER)
    assert isinstance(declared.strategy, Nvfp4PackedGateUp)
    assert not isinstance(plain.strategy, Nvfp4PackedGateUp)


def test_matches_gather_qmm() -> None:
    leaf = paired_leaf(seed=0)
    strategy = build(leaf)
    assert strategy is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    keys = shuffled_keys(seed=1)

    fused = strategy(x, mx.array(keys, dtype=mx.uint32))

    assert_matches(fused, swiglu_reference(x, leaf, selected(keys)))


def test_ties_resolve_to_the_lower_expert_index() -> None:
    leaf = paired_leaf(seed=2)
    strategy = build(leaf)
    assert strategy is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    keys = shuffled_keys(seed=3)
    # Two experts share the best ordinal: the lower index must take the first slot.
    keys[5] = 0
    keys[9] = 0
    assert selected(keys)[:2] == [5, 9]

    fused = strategy(x, mx.array(keys, dtype=mx.uint32))

    assert_matches(fused, swiglu_reference(x, leaf, selected(keys)))


def test_the_default_resolves_the_same_experts_from_the_keys() -> None:
    """`OrdinalRouting.indices` is the declaration's meaning spelled in ops, which is what
    keeps the delegator total when the packed kernel does not apply."""
    keys = shuffled_keys(seed=8)
    keys[5] = 0
    keys[9] = 0
    resolved = OrdinalRouting(TOPK).indices(mx.array(keys, dtype=mx.uint32))
    assert resolved.dtype == mx.uint32
    assert resolved.tolist() == selected(keys)


def test_patch_header_restores_the_two_unequal_spans() -> None:
    """The quantizer's first simdgroup writes the first span of a tensor twice, so expert
    0's gate row 0 and up row 0 may break the pairing. Their odd byte lives in the header
    and one lane per row reads it from there."""
    leaf = paired_leaf(seed=4)
    leaf.scales[0, 0, 1] = mx.array(0x40, dtype=mx.uint8)
    leaf.scales[0, 1, 1] = mx.array(0x41, dtype=mx.uint8)
    strategy = build(leaf)
    assert strategy is not None
    x = mx.random.normal((HIDDEN,)).astype(mx.bfloat16)
    keys = shuffled_keys(seed=5)
    keys[0] = 0  # expert 0 selected, so the patched rows are actually read

    fused = strategy(x, mx.array(keys, dtype=mx.uint32))

    assert_matches(fused, swiglu_reference(x, leaf, selected(keys)))


def test_build_refuses_an_unpaired_plane() -> None:
    leaf = paired_leaf(seed=6)
    leaf.scales[1, 3, 5] = mx.array(0x42, dtype=mx.uint8)
    assert build(leaf) is None


def test_packing_refuses_a_signed_scale_byte() -> None:
    """A byte with its sign bit set decodes through the wrong exponent, not as a negative
    number: the halved plane is only installable while the whole plane stays positive."""
    leaf = paired_leaf(seed=7)
    leaf.scales[2, 4, 6] = mx.array(0x80, dtype=mx.uint8)
    leaf.scales[2, 4, 7] = mx.array(0x80, dtype=mx.uint8)
    assert halved_group32_scales(leaf.scales, (0,)) is None
