"""The T=1 causal conv step in a single dispatch.

Unfused the step is a concat, `kernel` products, `kernel - 1` adds, a bias add, a
sigmoid and a product, plus the window slice — about nine dispatches over rows of a few
thousand channels. Here each thread owns one channel and runs the whole chain, rounding
to T at every boundary the op path rounds at: each tap product, each accumulation, the
bias add, the sigmoid, and the final product — so the fused output is bit-identical to
the op chain.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.core.mxcompat import metal_kernel

_SOURCE = """
    uint c = thread_position_in_grid.x;
    uint cd = (uint)CD;
    if (c >= cd) return;
    {
    #pragma clang fp contract(off)
    constexpr uint k = KERNEL;
    float m = (float)(T)((float)WIN[c] * (float)TAPS[c * k + 0]);
    for (uint t = 1; t + 1 < k; t++) {
        float p = (float)(T)((float)WIN[t * cd + c] * (float)TAPS[c * k + t]);
        m = (float)(T)(m + p);
    }
    float last = (float)(T)((float)X[c] * (float)TAPS[c * k + (k - 1)]);
    m = (float)(T)(m + last);
    if (HAS_BIAS) {
        m = (float)(T)(m + (float)BIAS[c]);
    }
    // mlx's own Sigmoid: 1/(1+exp(|x|)) mirrored by sign, with every op in T — the
    // stock unary kernel is templated on T and a bf16 chain rounds at each step.
    T mt = (T)m;
    T y = (T)1 / ((T)1 + metal::precise::exp(metal::abs(mt)));
    T s = (mt < (T)0) ? y : (T)1 - y;
    OUT[c] = (T)((T)m * s);
    for (uint t = 0; t + 2 < k; t++) {
        WOUT[t * cd + c] = WIN[(t + 1) * cd + c];
    }
    WOUT[(k - 2) * cd + c] = X[c];
    }
"""

_KERNEL = metal_kernel(
    name="conv_step",
    input_names=["X", "TAPS", "BIAS", "WIN", "CD"],
    output_names=["OUT", "WOUT"],
    source=_SOURCE,
)


@dataclass(frozen=True)
class FusedConvStep:
    taps: mx.array
    bias: mx.array
    conv_dim: int
    kernel: int
    has_bias: bool

    @classmethod
    def build(
        cls,
        *,
        taps: mx.array,
        bias: mx.array | None,
        conv_dim: int,
        kernel: int,
    ) -> Self | None:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
        if kernel < 2:
            return None
        # The kernel indexes BIAS unconditionally, so a missing bias becomes zeros the
        # HAS_BIAS branch never reads.
        filler = bias if bias is not None else mx.zeros((conv_dim,), dtype=taps.dtype)
        return cls(taps, filler, conv_dim, kernel, bias is not None)

    def __call__(self, x: mx.array, window: mx.array) -> tuple[mx.array, mx.array]:
        out = _KERNEL(
            inputs=[x, self.taps, self.bias, window, mx.array(self.conv_dim, mx.int32)],
            template=[
                ("T", x.dtype),
                ("KERNEL", self.kernel),
                ("HAS_BIAS", self.has_bias),
            ],
            grid=(self.conv_dim, 1, 1),
            threadgroup=(min(self.conv_dim, 256), 1, 1),
            output_shapes=[(self.conv_dim,), (self.kernel - 1, self.conv_dim)],
            output_dtypes=[x.dtype, x.dtype],
        )
        return out[0], out[1]
