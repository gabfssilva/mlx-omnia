"""The Mamba2 mixer's T=1 middle: one kernel module per shape, one delegator.

A mixer declares its conv, recurrence and norm leaves and their dimensions, and
`MambaStep` binds the specialization that declaration admits, once, at construction.
`fused.py` serves the group-per-threadgroup single dispatch; `default.py` serves
everything else through the existing `conv_step` + `ssm` + ops chain, so the delegator
is total.

`verify.py` is machinery around the step rather than a strategy: the same middle over
a verification round's few rows with per-token states, bound off a resolved
`FusedMambaStep`. `MambaStep.verify()` is the door to it — None when the resolved
strategy has no such form and the caller replays instead.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.mamba_step.default import DefaultMambaStep
from mlx_omnia.engine.core.kernels.mamba_step.fused import FusedMambaStep
from mlx_omnia.engine.core.kernels.mamba_step.kernel import MambaStepStrategy
from mlx_omnia.engine.core.kernels.mamba_step.verify import VerifyMambaStep
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = [
    "DefaultMambaStep",
    "FusedMambaStep",
    "MambaStep",
    "MambaStepStrategy",
    "VerifyMambaStep",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (FusedMambaStep, DefaultMambaStep)


class MambaStep(MambaStepStrategy):
    """Resolves the strategy at construction and delegates; itself a
    `MambaStepStrategy`."""

    def __init__(
        self,
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
    ) -> None:
        self.strategy: MambaStepStrategy = resolve(
            _STRATEGIES,
            taps=taps,
            conv_bias=conv_bias,
            A_log=A_log,
            D=D,
            dt_bias=dt_bias,
            norm_weight=norm_weight,
            eps=eps,
            inner=inner,
            conv_dim=conv_dim,
            kernel=kernel,
            heads=heads,
            head_dim=head_dim,
            groups=groups,
            state_size=state_size,
            time_step_limit=time_step_limit,
        )

    def __call__(
        self, proj: mx.array, window: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        return self.strategy(proj, window, state)

    def verify(self) -> VerifyMambaStep | None:
        """The per-token-checkpoint form of the same middle, or None when the resolved
        strategy has no such form and a verification round replays the layer instead."""
        middle = self.strategy
        return VerifyMambaStep.of(middle) if isinstance(middle, FusedMambaStep) else None
