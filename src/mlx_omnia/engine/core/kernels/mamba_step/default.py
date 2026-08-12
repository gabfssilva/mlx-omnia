"""The ordinary MLX chain for the Mamba2 T=1 middle; always builds.

A transcription of the mixer's own decode path between the projections — the same
`ConvStep` chain, the same fp32 softplus and clamp, the same `ssm_step` recurrence and
grouped gated RMSNorm — so the fused strategy has a parity reference whose rounding
boundaries are the model's own.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.conv_step import ConvStep
from mlx_omnia.engine.core.kernels.ssm import Ssm


@dataclass(frozen=True)
class DefaultMambaStep:
    conv: ConvStep
    scan: Ssm
    norm_weight: mx.array
    eps: float
    inner: int
    heads: int
    head_dim: int
    groups: int
    state_size: int

    @classmethod
    def build(
        cls,
        *,
        taps: mx.array,
        conv_bias: mx.array | None,
        A_log: mx.array,
        D: mx.array,
        dt_bias: mx.array,
        norm_weight: mx.array,
        eps: float,
        inner: int,
        conv_dim: int,
        kernel: int,
        heads: int,
        head_dim: int,
        groups: int,
        state_size: int,
        time_step_limit: tuple[float, float],
    ) -> Self | None:
        return cls(
            ConvStep(taps=taps, bias=conv_bias, conv_dim=conv_dim, kernel=kernel),
            Ssm(
                A_log=A_log,
                D=D,
                dt_bias=dt_bias,
                d_state=state_size,
                heads=heads,
                groups=groups,
                time_step_limit=time_step_limit,
            ),
            norm_weight,
            eps,
            inner,
            heads,
            head_dim,
            groups,
            state_size,
        )

    def __call__(
        self, proj: mx.array, window: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        inner = self.inner
        gate, conv_in, dt = mx.split(proj, [inner, proj.size - self.heads])
        mixed, slid = self.conv(conv_in, window)
        groups_state = self.groups * self.state_size
        hidden, b, c = mx.split(mixed, [inner, inner + groups_state])
        out, new_state = self.scan(
            hidden.reshape(1, 1, self.heads, self.head_dim),
            b.reshape(1, 1, self.groups, self.state_size),
            c.reshape(1, 1, self.groups, self.state_size),
            dt.reshape(1, 1, self.heads),
            state[None],
        )
        gated = (gate * mx.sigmoid(gate) * out.reshape(-1)).astype(mx.float32)
        grouped = mx.unflatten(gated, axis=-1, shape=(self.groups, -1))
        normed = mx.fast.rms_norm(grouped, None, self.eps).flatten(-2)
        return (
            self.norm_weight * normed.astype(proj.dtype),
            slid,
            new_state[0],
        )
