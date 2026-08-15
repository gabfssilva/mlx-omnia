"""The universal short-conv strategy: the step as the op chain it already is.

`build` accepts every shape, every kernel width and both biases, so it registers last
and makes the delegator total. The projection is a gemv rounded to the activation
dtype, B*x rounds again, the taps accumulate in float32 and round once before C's
gate — the boundaries the fused kernel reproduces. Its cost is the dispatches the
fused strategy collapses, not a different number.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.conv_mix.kernel import ConvMixStrategy


@dataclass(frozen=True)
class DefaultConvMix(ConvMixStrategy):
    hidden: int
    kernel: int
    proj_bias: mx.array | None
    conv_bias: mx.array | None

    @classmethod
    def build(
        cls,
        *,
        hidden: int,
        kernel: int,
        proj_bias: mx.array | None,
        conv_bias: mx.array | None,
    ) -> Self:
        return cls(hidden, kernel, proj_bias, conv_bias)

    def __call__(
        self, x: mx.array, weights: mx.array, taps: mx.array, window: mx.array
    ) -> tuple[mx.array, mx.array]:
        dtype = x.dtype
        projected = x[None] @ weights.T
        if self.proj_bias is not None:
            projected = projected + self.proj_bias
        b, c, v = mx.split(projected.astype(dtype), 3, axis=-1)
        padded = mx.concatenate([window, b * v])

        lifted = padded.astype(mx.float32)
        tap = taps.reshape(self.hidden, self.kernel)
        conv = lifted[0] * tap[:, 0]
        for j in range(1, self.kernel):
            conv = conv + lifted[j] * tap[:, j]
        if self.conv_bias is not None:
            conv = conv + self.conv_bias
        return (c[0] * conv.astype(dtype)), padded[1:]
