"""The plain causal short conv's T=1 step: one kernel module per shape, one delegator.

A mixer declares the conv — the taps, the optional bias, the channel count and kernel
width — and `ConvStep` binds the specialization that declaration admits, once, at
construction. `fused.py` serves any kernel width of at least 2 on the GPU; `default.py`
serves everything else through the op chain, so the delegator is total.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.conv_step.default import DefaultConvStep
from mlx_omnia.engine.core.kernels.conv_step.fused import FusedConvStep
from mlx_omnia.engine.core.kernels.conv_step.kernel import ConvStepStrategy
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = ["ConvStep", "ConvStepStrategy", "DefaultConvStep", "FusedConvStep"]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (FusedConvStep, DefaultConvStep)


class ConvStep:
    """Resolves the strategy at construction and delegates; itself a
    `ConvStepStrategy`."""

    def __init__(
        self,
        *,
        taps: mx.array,
        bias: mx.array | None,
        conv_dim: int,
        kernel: int,
    ) -> None:
        self.strategy: ConvStepStrategy = resolve(
            _STRATEGIES,
            taps=taps,
            bias=bias,
            conv_dim=conv_dim,
            kernel=kernel,
        )

    def __call__(self, x: mx.array, window: mx.array) -> tuple[mx.array, mx.array]:
        return self.strategy(x, window)
