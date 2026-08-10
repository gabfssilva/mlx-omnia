"""The SSD scan's contract: what a strategy is and the dt preprocessing they share.

The primitive: the selective scan advanced over `T` tokens,
(hidden [B, T, H, Dh], b [B, T, G, Ds], c [B, T, G, Ds], dt [B, T, H], state
[B, H, Dh, Ds] fp32) -> (out [B, T, H, Dh] in `hidden`'s dtype, advanced state).
`dt` arrives pre-softplus; `compute_dt` is the host side every strategy applies
before its own arithmetic. The `Ssm` delegator in `__init__.py` resolves which
strategy serves a given configuration, once, at construction.
"""

from typing import Protocol

import mlx.core as mx
import mlx.nn as nn


class SsmStrategy(Protocol):
    """The SSD scan over `T` tokens, carrying the state forward."""

    def __call__(
        self,
        hidden: mx.array,
        b: mx.array,
        c: mx.array,
        dt: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]: ...


def compute_dt(
    dt: mx.array, dt_bias: mx.array, time_step_limit: tuple[float, float]
) -> mx.array:
    """softplus(dt + dt_bias) in fp32, clamped to the time-step limit."""
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias.astype(mx.float32))
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])
