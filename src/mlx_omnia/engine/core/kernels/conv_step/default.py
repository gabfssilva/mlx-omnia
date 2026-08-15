"""The ordinary MLX chain for the T=1 causal conv step; always builds.

The op order is a transcription of the mamba mixer's `convolve` at length 1 — product,
adds, bias, SiLU, each rounding to the input dtype — so the fused strategy has a parity
reference whose rounding boundaries are the model's own.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.conv_step.kernel import ConvStepStrategy


@dataclass(frozen=True)
class DefaultConvStep(ConvStepStrategy):
    taps: mx.array
    bias: mx.array | None

    @classmethod
    def build(
        cls,
        *,
        taps: mx.array,
        bias: mx.array | None,
        conv_dim: int,
        kernel: int,
    ) -> Self | None:
        return cls(taps, bias)

    def __call__(self, x: mx.array, window: mx.array) -> tuple[mx.array, mx.array]:
        padded = mx.concatenate([window, x[None]], axis=0)
        mixed = padded[0] * self.taps[:, 0]
        for tap in range(1, self.taps.shape[1]):
            mixed = mixed + padded[tap] * self.taps[:, tap]
        if self.bias is not None:
            mixed = mixed + self.bias
        return mixed * mx.sigmoid(mixed), padded[1:]
