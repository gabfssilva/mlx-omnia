"""The universal SSD strategy: the recurrence unrolled token by token in ops.

`build` accepts every configuration, so it registers last and makes the delegator
total — `Ssm` always resolves. The scan runs entirely in float32, one iteration per
token, materializing the group broadcast with `repeat`: `dA = exp(A·dt)`,
`state = dA·state + dt·B·x`, `y = state·C + D·x`. What defines it is universality,
not the absence of a kernel; it is also the parity reference the decode kernel is
tested against, and its cost is the per-token dispatch chain the specialized
strategy fuses away.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.ssm.kernel import SsmStrategy, compute_dt


def ssm_step_ref(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: tuple[float, float],
) -> tuple[mx.array, mx.array]:
    """The SSD recurrence, token by token, entirely in float32 — the parity
    reference for the decode kernel.

    `x` is `[B, T, H, Dh]`, `B`/`C` are `[B, T, G, Ds]`, `dt` is `[B, T, H]`
    (pre-softplus), `state` is `[B, H, Dh, Ds]`. Returns the output `[B, T, H, Dh]`
    in `x`'s dtype and the advanced state.
    """
    _batch, length, num_heads, _head_dim = x.shape
    n_groups = B.shape[2]
    repeats = num_heads // n_groups

    A = -mx.exp(A_log.astype(mx.float32))
    dt_processed = compute_dt(dt, dt_bias, time_step_limit)
    dA = mx.exp(dt_processed[..., None] * A[None, None, :, None])
    x32 = x.astype(mx.float32)

    B_expanded = mx.repeat(B, repeats, axis=2)
    C_expanded = mx.repeat(C, repeats, axis=2)

    out_list: list[mx.array] = []
    for t in range(length):
        dA_t = dA[:, t]
        dB = dt_processed[:, t, :, None, None] * B_expanded[:, t, :, None, :].astype(
            mx.float32
        )
        dBx = dB * x32[:, t, :, :, None]
        state = state * dA_t[:, :, :, None] + dBx
        y_t = (state * C_expanded[:, t, :, None, :].astype(mx.float32)).sum(axis=-1)
        y_t = y_t + x32[:, t] * D[None, :, None].astype(mx.float32)
        out_list.append(y_t)

    out = mx.stack(out_list, axis=1) if out_list else mx.zeros_like(x32)
    return out.astype(x.dtype), state


@dataclass(frozen=True)
class DefaultSsm(SsmStrategy):
    A_log: mx.array
    D: mx.array
    dt_bias: mx.array
    time_step_limit: tuple[float, float]

    @classmethod
    def build(
        cls,
        *,
        A_log: mx.array,
        D: mx.array,
        dt_bias: mx.array,
        d_state: int,
        heads: int,
        groups: int,
        time_step_limit: tuple[float, float],
        step: int,
    ) -> Self:
        return cls(A_log, D, dt_bias, time_step_limit)

    def __call__(
        self,
        hidden: mx.array,
        b: mx.array,
        c: mx.array,
        dt: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return ssm_step_ref(
            hidden, self.A_log, b, c, self.D, dt, self.dt_bias, state,
            self.time_step_limit,
        )
