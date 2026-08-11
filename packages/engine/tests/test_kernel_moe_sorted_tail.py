"""Parity for the sorted MoE tail against the gather-then-combine it replaces.

There is no quantization here and no reassociation: the kernel reads the same rows the
un-sort would have produced, in the same slot order, and rounds at the same places. The
reference is the default strategy — the explicit gather followed by the same bf16 chain —
so the two agree to within one bf16 ulp; a wrong `inverse_order` read, by contrast, is a
different row.
"""

import mlx.core as mx
from conftest import relative_diff

from mlx_omnia.core.kernels.moe_tail import DefaultMoeTail, MoeTail, SortedMoeTail

TOKENS = 3
TOPK = 4
HIDDEN = 1024
SCALING = 2.5
SLACK = 2.0**-8


def assert_matches(ours: mx.array, expected: mx.array) -> None:
    """A zero reference makes both sides identically zero, and the house metric divides by
    the reference's magnitude — there is no relative bound to take there."""
    if float(mx.max(mx.abs(expected))) == 0.0:
        assert float(mx.max(mx.abs(ours))) == 0.0
        return
    assert relative_diff(ours, expected) < SLACK


def test_delegator_prefers_the_kernel_and_stays_total() -> None:
    assert isinstance(MoeTail(hidden=HIDDEN).strategy, SortedMoeTail)
    assert isinstance(MoeTail(hidden=HIDDEN + 2).strategy, DefaultMoeTail)


def test_matches_gather_then_combine() -> None:
    mx.random.seed(0)
    sorted_outputs = mx.random.normal((TOKENS * TOPK, HIDDEN)).astype(mx.bfloat16)
    # A permutation: every sorted row is claimed by exactly one (token, slot).
    order = [(index * 7 + 5) % (TOKENS * TOPK) for index in range(TOKENS * TOPK)]
    assert sorted(order) == list(range(TOKENS * TOPK))
    inverse_order = mx.array(order, dtype=mx.uint32).reshape(TOKENS, TOPK)
    routing = mx.random.uniform(shape=(TOKENS, TOPK)).astype(mx.float32)
    shared = mx.random.normal((TOKENS, HIDDEN)).astype(mx.bfloat16)
    residual = mx.random.normal((TOKENS, HIDDEN)).astype(mx.bfloat16)

    args = (sorted_outputs, inverse_order, routing, shared, residual, SCALING)
    fused = MoeTail(hidden=HIDDEN)(*args)

    assert_matches(fused, DefaultMoeTail.build(hidden=HIDDEN)(*args))
