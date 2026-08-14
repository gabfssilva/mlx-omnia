"""The SSD selective scan: one strategy per implementation, one delegator.

A mixer declares the scan — the head/state geometry, the per-head `A_log`, `D` and
`dt_bias`, the time-step limit and the prefill chunk — and `Ssm` binds the fused
path when the kernel's tiling divides that geometry, or the ops scan when it does
not, at construction time. The mixer never names an implementation.

`fused.py` carries both regimes behind one call (the one-dispatch decode kernel from
`step.py`, the chunked prefill from `chunked.py`); `default.py` serves everything
else with the token-by-token float32 recurrence, which is also the parity reference,
so the delegator is total: a model uses `Ssm` like any other layer. The free
functions the mixers call directly — `ssm_step`, `ssm_attn`, `ssm_update`,
`ssm_step_ref`, `ssm_applies` — remain exported here.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.resolve import resolve
from mlx_omnia.engine.core.kernels.ssm.chunked import ssm_applies, ssm_attn, ssm_step, ssm_update
from mlx_omnia.engine.core.kernels.ssm.default import DefaultSsm, ssm_step_ref
from mlx_omnia.engine.core.kernels.ssm.fused import FusedSsm
from mlx_omnia.engine.core.kernels.ssm.kernel import SsmStrategy, compute_dt

__all__ = [
    "DefaultSsm",
    "FusedSsm",
    "Ssm",
    "SsmStrategy",
    "compute_dt",
    "ssm_applies",
    "ssm_attn",
    "ssm_step",
    "ssm_step_ref",
    "ssm_update",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (FusedSsm, DefaultSsm)


class Ssm:
    """Resolves the strategy at construction and delegates; itself an
    `SsmStrategy`."""

    def __init__(
        self,
        *,
        A_log: mx.array,
        D: mx.array,
        dt_bias: mx.array,
        d_state: int,
        heads: int,
        groups: int,
        time_step_limit: tuple[float, float] = (0.0, float("inf")),
        step: int = 256,
    ) -> None:
        self.strategy: SsmStrategy = resolve(
            _STRATEGIES,
            A_log=A_log,
            D=D,
            dt_bias=dt_bias,
            d_state=d_state,
            heads=heads,
            groups=groups,
            time_step_limit=time_step_limit,
            step=step,
        )

    def __call__(
        self,
        hidden: mx.array,
        b: mx.array,
        c: mx.array,
        dt: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self.strategy(hidden, b, c, dt, state)
