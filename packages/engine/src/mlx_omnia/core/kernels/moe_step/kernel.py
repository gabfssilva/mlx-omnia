"""The gate-free sparse block's T=1 step contract.

The primitive: one token row through its k routed experts — `fc1`, squared ReLU,
`fc2` — the router-weighted combine, and the unrouted (shared) expert's output added at
the end when the block declares one —
(x [hidden], chosen [k] uint32, weights [k] fp32 already carrying the routed scaling)
-> [hidden]. The gate is not here: which experts and at what weight is the router's
answer, produced upstream. The shared expert *is* here, as construction-time leaves:
its chain is part of the same tail, and a strategy that can ride it in the same
dispatches saves the ops the caller would otherwise pay. The `MoeStep` delegator in
`__init__.py` resolves which module serves a given declaration, once, at construction.
"""

from typing import Protocol

import mlx.core as mx


class MoeStepStrategy(Protocol):
    """(x [hidden], chosen [k], weights [k]) ->
    `(Σ_e weights[e] · fc2_e(squared_relu(fc1_e x))) + shared(x)`, rounded the way the
    op chain rounds: each projection to the input dtype, the combine accumulated in
    fp32 and cast once, the shared add in the input dtype."""

    def __call__(self, x: mx.array, chosen: mx.array, weights: mx.array) -> mx.array: ...
