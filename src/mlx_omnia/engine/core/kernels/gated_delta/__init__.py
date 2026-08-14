"""The gated delta rule: one fused kernel, the ops recurrence, one delegator.

A linear-attention layer declares its head geometry and `GatedDelta` binds the fused
kernel when its tiling admits those shapes and the layer's A/B switch is on, or the ops
recurrence otherwise, at construction time. Both halves speak the kernel's convention
(`kernel.py`), so the model writes one call: the decay past the exp, `q` pre-scaled,
q/k unrepeated, the state `[B, Hv, Dv, Dk]`.

`delta_rule` stays exported for the callers that walk the recurrence in the model's own
convention — the log decay and a `[B, Hv, Dk, Dv]` state.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.gated_delta.default import DefaultGatedDelta, delta_rule
from mlx_omnia.engine.core.kernels.gated_delta.fused import (
    FusedGatedDelta,
    gated_delta,
    gated_delta_applies,
)
from mlx_omnia.engine.core.kernels.gated_delta.kernel import GatedDeltaStrategy
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = [
    "DefaultGatedDelta",
    "FusedGatedDelta",
    "GatedDelta",
    "GatedDeltaStrategy",
    "delta_rule",
    "gated_delta",
    "gated_delta_applies",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (FusedGatedDelta, DefaultGatedDelta)


class GatedDelta:
    """Resolves the strategy at construction and delegates; itself a
    `GatedDeltaStrategy`."""

    def __init__(
        self,
        *,
        key_dim: int,
        key_heads: int,
        value_heads: int,
        value_dim: int,
        enabled: bool = True,
    ) -> None:
        self.strategy: GatedDeltaStrategy = resolve(
            _STRATEGIES,
            key_dim=key_dim,
            key_heads=key_heads,
            value_heads=value_heads,
            value_dim=value_dim,
            enabled=enabled,
        )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self.strategy(q, k, v, g, beta, state)
