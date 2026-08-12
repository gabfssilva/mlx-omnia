"""The fused SSD strategy: the one-dispatch decode kernel plus the chunked prefill.

Both halves of what the scan needs, behind one call: T=1 goes to the Metal step
(`step.py`, the whole recurrence in a single dispatch, the group broadcast done
in-kernel), anything longer goes to the chunked surrogate-attention prefill
(`chunked.py`, matmul-shaped ops instead of a per-token chain). That is the split
`ssm_update` already dispatches; the strategy only resolves the shape predicate
once instead of on every token.

`build` declines when the kernel's tiling does not divide the configuration, or
when the default device is not the GPU — the ops scan in `default.py` serves those.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.ssm.chunked import ssm_attn
from mlx_omnia.engine.core.kernels.ssm.step import ssm_step, ssm_step_applies

_SMALL = 8


@dataclass(frozen=True)
class FusedSsm:
    A_log: mx.array
    D: mx.array
    dt_bias: mx.array
    time_step_limit: tuple[float, float]
    step: int

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
    ) -> Self | None:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
        if not ssm_step_applies(d_state, heads, groups):
            return None
        return cls(A_log, D, dt_bias, time_step_limit, step)

    def __call__(
        self,
        hidden: mx.array,
        b: mx.array,
        c: mx.array,
        dt: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        length = hidden.shape[1]
        if length <= _SMALL:
            # A speculative verification hands over two to four rows, and the chunked
            # scan's cost is flat in T — pure structure, measured ~1.7x the iterated
            # step at T=3 across a trunk's 23 layers. Iterating the decode kernel also
            # keeps the verification's arithmetic the decode's own.
            outs: list[mx.array] = []
            for position in range(length):
                out, state = ssm_step(
                    hidden[:, position : position + 1],
                    self.A_log,
                    b[:, position : position + 1],
                    c[:, position : position + 1],
                    self.D,
                    dt[:, position : position + 1],
                    self.dt_bias,
                    state,
                    self.time_step_limit,
                )
                outs.append(out)
            return (outs[0] if length == 1 else mx.concatenate(outs, axis=1)), state
        return ssm_attn(
            hidden, self.A_log, b, c, self.D, dt, self.dt_bias, state,
            time_step_limit=self.time_step_limit, step=self.step,
        )
