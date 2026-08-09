"""Parity for the sorted MoE tail against the gather-then-combine it replaces.

There is no quantization here and no reassociation: the kernel reads the same rows the
un-sort would have produced, in the same slot order, and rounds at the same places. The
reference is the explicit gather followed by the same bf16 chain, so the two agree to
within one bf16 ulp — a wrong `inverse_order` read, by contrast, is a different row.
"""

import mlx.core as mx
from conftest import relative_diff

from sideros.core.kernels.moe_sorted_tail import moe_sorted_tail, moe_sorted_tail_applies

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


def tail_reference(
    sorted_outputs: mx.array,
    inverse_order: mx.array,
    routing: mx.array,
    shared: mx.array,
    residual: mx.array,
) -> mx.array:
    gathered = mx.take(sorted_outputs, inverse_order.reshape(-1), axis=0).reshape(
        TOKENS, TOPK, HIDDEN
    )
    weights = routing.astype(mx.bfloat16)
    total = mx.zeros((TOKENS, HIDDEN), dtype=mx.bfloat16)
    for slot in range(TOPK):
        total = gathered[:, slot] * weights[:, slot : slot + 1] + total
    scaled = total * mx.array(SCALING, dtype=mx.bfloat16)
    return residual + (scaled + shared)


def test_applies_requires_four_column_tiles() -> None:
    assert moe_sorted_tail_applies(HIDDEN)
    assert not moe_sorted_tail_applies(HIDDEN + 2)


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

    fused = moe_sorted_tail(
        sorted_outputs, inverse_order, routing, shared, residual, SCALING
    )

    assert_matches(
        fused, tail_reference(sorted_outputs, inverse_order, routing, shared, residual)
    )
