"""The gated short conv's one-token step in a single dispatch.

Unfused the step is a gemv plus the elementwise chain that splits B, C and x,
multiplies B by x, applies the three causal taps against the two cached rows,
gates by C and slides the window. Every rounding boundary of that chain is
reproduced here: the projections round to T before the product, the product
rounds to T before the float32 tap accumulation, and the conv rounds to T
before the gate — so the fused output is bit-identical to the op path once the
projections agree.

The in_proj weight comes in checkpoint layout [3*hidden, hidden] with the B, C, x
blocks stacked in that order; the taps are the conv weight flattened to
[hidden*3]; the window holds the previous two B*x rows.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.conv_mix.kernel import ConvMixStrategy
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint pair = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint c0 = pair * 2;
    uint h = (uint)HD;
    uint k4 = h / 4;
    if (c0 >= h) return;

    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;
    const device vec<T, 4>* wB0 = (const device vec<T, 4>*)(W + (size_t)c0 * h);
    const device vec<T, 4>* wB1 = (const device vec<T, 4>*)(W + (size_t)(c0 + 1) * h);
    const device vec<T, 4>* wC0 = (const device vec<T, 4>*)(W + (size_t)(h + c0) * h);
    const device vec<T, 4>* wC1 = (const device vec<T, 4>*)(W + (size_t)(h + c0 + 1) * h);
    const device vec<T, 4>* wX0 = (const device vec<T, 4>*)(W + (size_t)(2 * h + c0) * h);
    const device vec<T, 4>* wX1 = (const device vec<T, 4>*)(W + (size_t)(2 * h + c0 + 1) * h);

    float4 aB0 = float4(0.0f), aC0 = float4(0.0f), aX0 = float4(0.0f);
    float4 aB1 = float4(0.0f), aC1 = float4(0.0f), aX1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xm = float4(xv[i]);
        aB0 += xm * float4(wB0[i]); aB1 += xm * float4(wB1[i]);
        aC0 += xm * float4(wC0[i]); aC1 += xm * float4(wC1[i]);
        aX0 += xm * float4(wX0[i]); aX1 += xm * float4(wX1[i]);
    }
    float B[2], C[2], Xp[2];
    B[0] = simd_sum(aB0.x + aB0.y + aB0.z + aB0.w);
    B[1] = simd_sum(aB1.x + aB1.y + aB1.z + aB1.w);
    C[0] = simd_sum(aC0.x + aC0.y + aC0.z + aC0.w);
    C[1] = simd_sum(aC1.x + aC1.y + aC1.z + aC1.w);
    Xp[0] = simd_sum(aX0.x + aX0.y + aX0.z + aX0.w);
    Xp[1] = simd_sum(aX1.x + aX1.y + aX1.z + aX1.w);
    if (lane == 0) {
        // Without this, Metal contracts the tap accumulation into fmas and the float32
        // conv lands ~1 ulp off the op chain's separate multiply and add.
        #pragma clang fp contract(off)
        for (uint r = 0; r < 2; r++) {
            uint c = c0 + r;
            float bx = (float)(T)((float)(T)B[r] * (float)(T)Xp[r]);
            float conv = (float)TAPS[c * 3 + 0] * (float)WIN[c];
            conv = conv + (float)TAPS[c * 3 + 1] * (float)WIN[h + c];
            conv = conv + (float)TAPS[c * 3 + 2] * bx;
            GATED[c] = (T)((float)(T)C[r] * (float)(T)conv);
            WOUT[c] = WIN[h + c];
            WOUT[h + c] = (T)bx;
        }
    }
"""

_KERNEL = metal_kernel(
    name="conv_mix",
    input_names=["X", "W", "TAPS", "WIN", "HD"],
    output_names=["GATED", "WOUT"],
    source=_SOURCE,
)


def applies(hidden: int, kernel: int, has_bias: bool) -> bool:
    """Kernel 3 without a conv bias, over a channel count the float4 load tiles."""
    return kernel == 3 and not has_bias and hidden % 4 == 0


@dataclass(frozen=True)
class FusedConvMix(ConvMixStrategy):
    hidden: int

    @classmethod
    def build(
        cls,
        *,
        hidden: int,
        kernel: int,
        proj_bias: mx.array | None,
        conv_bias: mx.array | None,
    ) -> Self | None:
        if proj_bias is not None or conv_bias is not None:
            return None
        if not applies(hidden, kernel, has_bias=False):
            return None
        return cls(hidden)

    def __call__(
        self, x: mx.array, weights: mx.array, taps: mx.array, window: mx.array
    ) -> tuple[mx.array, mx.array]:
        hidden = self.hidden
        assert x.ndim == 1 and weights.shape == (3 * hidden, hidden)
        assert taps.shape == (3 * hidden,) and window.shape == (2, hidden)
        out = _KERNEL(
            inputs=[x, weights, taps, window, mx.array(hidden, mx.int32)],
            template=[("T", x.dtype)],
            grid=(32, hidden // 2, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(hidden,), (2, hidden)],
            output_dtypes=[x.dtype, x.dtype],
        )
        return out[0], out[1]
