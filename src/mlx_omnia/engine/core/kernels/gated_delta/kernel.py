"""The gated delta rule's contract: what a strategy is and what a model declares.

The primitive: the recurrence walked over the whole sequence —
(q [B, T, Hk, Dk] l2-normalized and **already scaled**, k [B, T, Hk, Dk] l2-normalized,
v [B, T, Hv, Dv], g the decay **past the exp** — `[B, T, Hv]` per head or
`[B, T, Hv, Dk]` per key channel —, beta [B, T, Hv], state [B, Hv, Dv, Dk] float32) ->
(y [B, T, Hv, Dv], state). The key heads are broadcast into the value heads by the
strategy, so no repeated q/k is ever materialized by the caller.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class GatedDeltaStrategy(Protocol):
    """(q, k, v, g, beta, state) -> (y, advanced state), in the kernel's convention."""

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]: ...
