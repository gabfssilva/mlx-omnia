"""The mxfp4 down/combine kernel. The e2m1 lookup and the shape predicate are shared
with the gate-up half, from `shared.mxfp4` — same bytes on both sides of the step.
The per-row projection bias is folded before the routing weight; no spare slot."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.down_combine.kernel import DownCombineStrategy, Layout
from mlx_omnia.engine.core.kernels.shared.mxfp4 import HEADER, applies
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    threadgroup float MXLUT[16];
    mxfp4Lut(MXLUT, tid);

    constexpr uint block = 8 * 32;

    uint kdim = (uint)KD;
    uint kw = kdim / 2;
    uint kg = kdim / 32u;
    uint rows = (uint)N;
    uint row0 = tgy * 16 + sg * 4;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint e = 0; e < (uint)TOPK; e++) {
        size_t wbase = (size_t)IDX[e] * rows;
        const device T* xe = ACT + e * kdim;
        float wt = (float)WTS[e];
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint base = lane * 8, blk = 0; blk < kdim; blk += block, base += block) {
            if (base >= kdim) continue;
            float xt[8];
            for (uint j = 0; j < 8; j++) xt[j] = (float)xe[base + j];
            for (uint row = 0; row < 4; row++) {
                uint q = *((const device uint*)((const device uint8_t*)W
                    + (wbase + row0 + row) * kw + base / 2));
                float s = as_type<float>((uint)S[(wbase + row0 + row) * kg + base / 32] << 23);
                res[row] += mxfp4Dot8(q, s, xt, MXLUT);
            }
        }
        for (uint row = 0; row < 4; row++) {
            float t = simd_sum(res[row]);
            if (lane == 0) {
                float tb = (float)(T)(t + (float)Bs[wbase + row0 + row]);
                acc[row] += (float)(T)(tb * wt);
            }
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < 4; row++) {
            uint i = row0 + row;
            Y[i] = (T)((float)(T)acc[row] + (float)RES[i]);
        }
    }
"""

_KERNEL = metal_kernel(
    name="mxfp4_down_combine",
    input_names=["ACT", "W", "S", "Bs", "IDX", "WTS", "RES", "N", "KD"],
    output_names=["Y"],
    source=_SOURCE,
    header=HEADER,
)


@dataclass(frozen=True)
class Mxfp4DownCombine(DownCombineStrategy):
    weight: mx.array
    scales: mx.array
    bias: mx.array

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
        if layout != "interleaved":
            return None
        # The kernel bakes the per-row projection bias and has no spare slot.
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "mxfp4":
            return None
        if bias is None or shared is not None:
            return None
        if not applies(hidden, inner):
            return None
        return cls(leaf.weight, leaf.scales, bias)

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array:
        hidden = self.weight.shape[1]
        kdim = self.weight.shape[2] * 8
        assert kdim % 32 == 0 and hidden % 16 == 0
        return _KERNEL(
            inputs=[
                act, self.weight, self.scales, self.bias, chosen, weights, residual,
                mx.array(hidden, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
            ],
            template=[("T", act.dtype), ("TOPK", chosen.shape[0])],
            grid=(128, hidden // 16, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(hidden,)],
            output_dtypes=[act.dtype],
        )[0]
