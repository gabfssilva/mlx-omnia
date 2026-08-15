"""The dense (unquantized) routed down projection, with the SwiGLU it consumes fused into
its own prologue.

`DenseGateUp`'s other half: under the `blocked` layout the activation block arrives
un-activated, `[k, 2*inner]` with gate and up block-concatenated, and silu(gate)·up is
computed while the down weights stream, so nothing elementwise runs between the two
dispatches. The routing weight folds into the output; the expert sum and the residual add
are ops, which is what the quantized strategies fuse and this one does not.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.down_combine.kernel import DownCombineStrategy, Layout
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint pair = thread_position_in_grid.y;
    uint expert = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint kd = (uint)KD;
    uint k4 = kd / 4;
    if (n0 >= n) return;

    size_t base = (size_t)IDX[expert] * n * kd;
    const device vec<T, 4>* w0 = (const device vec<T, 4>*)(W + base + (size_t)n0 * kd);
    const device vec<T, 4>* w1 = (const device vec<T, 4>*)(W + base + (size_t)(n0 + 1) * kd);
    const device vec<T, 4>* gv = (const device vec<T, 4>*)(GU + (size_t)expert * 2 * kd);
    const device vec<T, 4>* uv = (const device vec<T, 4>*)(GU + ((size_t)expert * 2 + 1) * kd);

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 g = float4(gv[i]);
        float4 xm = g / (1.0f + metal::exp(-g)) * float4(uv[i]);
        acc0 += xm * float4(w0[i]);
        acc1 += xm * float4(w1[i]);
    }
    float t0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float t1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0) {
        float scale = (float)WS[expert];
        Y[(size_t)expert * n + n0] = (T)(t0 * scale);
        Y[(size_t)expert * n + n0 + 1] = (T)(t1 * scale);
    }
"""

_KERNEL = metal_kernel(
    name="moe_dense_down",
    input_names=["GU", "W", "IDX", "WS", "N", "KD"],
    output_names=["Y"],
    source=_SOURCE,
)


def applies(hidden: int, inner: int) -> bool:
    """The contraction over `inner` is a float4 lane sweep, and the output rows are walked
    in pairs."""
    return inner % 4 == 0 and hidden % 2 == 0


@dataclass(frozen=True)
class DenseDownCombine(DownCombineStrategy):
    weight: mx.array

    @classmethod
    def build(
        cls,
        leaf: SwitchLinear | QuantizedSwitchLinear,
        *,
        hidden: int,
        inner: int,
        bias: mx.array | None,
        shared: nn.Linear | nn.QuantizedLinear | None,
        layout: Layout,
    ) -> Self | None:
        if not isinstance(leaf, SwitchLinear) or layout != "blocked":
            return None
        if bias is not None or shared is not None:
            return None
        if leaf.weight.shape[1:] != (hidden, inner) or not applies(hidden, inner):
            return None
        return cls(leaf.weight)

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array:
        rows = self.weight.shape[1]
        kdim = self.weight.shape[2]
        assert kdim % 4 == 0 and rows % 2 == 0
        routed = _KERNEL(
            inputs=[
                act, self.weight, chosen, weights,
                mx.array(rows, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
            ],
            template=[("T", act.dtype)],
            grid=(32, rows // 2, chosen.shape[0]),
            threadgroup=(32, 1, 1),
            output_shapes=[(chosen.shape[0], rows)],
            output_dtypes=[act.dtype],
        )[0]
        return routed.sum(axis=0) + residual
