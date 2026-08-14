"""The gated short conv's T=1 step: one kernel module per shape, one delegator.

A model block declares the conv — hidden width, kernel width, the optional projection
and conv biases — and `ConvMix` binds the specialization that declaration admits, or
none, at construction time. `fused.py` serves the bias-free kernel-3 conv over a
channel count the float4 load tiles; `default.py` serves everything else through the
op chain, so the delegator is total: a model uses `ConvMix` like any other layer.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.conv_mix.default import DefaultConvMix
from mlx_omnia.engine.core.kernels.conv_mix.fused import FusedConvMix
from mlx_omnia.engine.core.kernels.conv_mix.kernel import ConvMixStrategy
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = ["ConvMix", "ConvMixStrategy", "DefaultConvMix", "FusedConvMix"]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (FusedConvMix, DefaultConvMix)


class ConvMix:
    """Resolves the strategy at construction and delegates; itself a
    `ConvMixStrategy`."""

    def __init__(
        self,
        *,
        hidden: int,
        kernel: int,
        proj_bias: mx.array | None = None,
        conv_bias: mx.array | None = None,
    ) -> None:
        self.strategy: ConvMixStrategy = resolve(
            _STRATEGIES,
            hidden=hidden,
            kernel=kernel,
            proj_bias=proj_bias,
            conv_bias=conv_bias,
        )

    def __call__(
        self, x: mx.array, weights: mx.array, taps: mx.array, window: mx.array
    ) -> tuple[mx.array, mx.array]:
        return self.strategy(x, weights, taps, window)
