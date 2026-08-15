"""The gated sparse block's T=1 step contract.

The primitive: one token through sigmoid routing, its k routed SwiGLU experts, the
scaled combine, the shared expert, and the residual join —
(x [..., hidden], residual [..., hidden]) -> [..., hidden]. Routing is inside, unlike
`moe_step`: the packed strategy selects experts within its gate/up dispatch from the
router's ordinal keys, so which experts and at what weight is this primitive's answer,
not an operand. The shared expert is construction-time leaves; it quantizes on its own
schedule, so no declaration ties its format to the routed stack's.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class SwigluMoeStepStrategy(Protocol):
    """(x, residual, logits?, keys?) ->
    `residual + scale · Σ_e w_e · down_e(silu(gate_e x) · up_e x) + shared(x)`, rounded
    the way the op chain rounds.

    `logits` short-circuits the router gemv when a fused residual/router dispatch
    already produced the row's logits; `keys` does the same for the ordinal sort keys a
    routing-inside strategy would otherwise derive from them.
    """

    @property
    def worthwhile(self) -> bool:
        """False when nothing beyond the defaults answered — the caller's signal to
        stay on its batched forward instead of stepping token by token."""
        ...

    def __call__(
        self,
        x: mx.array,
        residual: mx.array,
        *,
        logits: mx.array | None = None,
        keys: mx.array | None = None,
    ) -> mx.array: ...

    def batch(self, h: mx.array, residual: mx.array) -> mx.array | None:
        """The same block over a batch of rows, or `None` when this strategy has no
        batched form and the caller should run its own forward."""
        ...
