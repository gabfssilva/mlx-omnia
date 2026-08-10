"""The nvfp4 down/combine kernel. The e4m3/e2m1 helpers and the shape predicate are
shared with the gate-up half, from `shared.nvfp4.qmoe` — same bytes on both sides of
the step. No spare slot and no projection bias."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.down_combine.kernel import Layout
from sideros.core.kernels.shared.nvfp4.qmoe import HEADER, applies
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear
from sideros.core.mxcompat import metal_kernel

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    threadgroup float NVLUT[16];
    nvfp4Lut(NVLUT, tid);

    constexpr uint block = 16 * 32;

    uint kdim = (uint)KD;
    uint kw = kdim / 2;
    uint kg = kdim / 16u;
    uint rows = (uint)N;
    uint row0 = tgy * 16 + sg * 4;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint e = 0; e < (uint)TOPK; e++) {
        size_t wbase = (size_t)IDX[e] * rows;
        const device T* xe = ACT + e * kdim;
        float wt = (float)WTS[e];
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint base = lane * 16, blk = 0; blk < kdim; blk += block, base += block) {
            if (base >= kdim) continue;
            float xt[16];
            for (uint j = 0; j < 16; j++) xt[j] = (float)xe[base + j];
            for (uint row = 0; row < 4; row++) {
                size_t r = wbase + row0 + row;
                uint2 q = *((const device uint2*)((const device uint8_t*)W + r * kw + base / 2));
                float s = nvfp4Scale(S[r * kg + base / 16]);
                res[row] += nvfp4Dot16(q, s, xt, NVLUT);
            }
        }
        for (uint row = 0; row < 4; row++) {
            float t = simd_sum(res[row]);
            if (lane == 0) {
                acc[row] += (float)(T)((float)(T)t * wt);
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
    name="nvfp4_down_combine",
    input_names=["ACT", "W", "S", "IDX", "WTS", "RES", "N", "KD"],
    output_names=["Y"],
    source=_SOURCE,
    header=HEADER,
)


@dataclass(frozen=True)
class Nvfp4DownCombine:
    weight: mx.array
    scales: mx.array

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
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "nvfp4":
            return None
        if bias is not None or shared is not None:
            return None
        if not applies(hidden, inner):
            return None
        return cls(leaf.weight, leaf.scales)

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array:
        hidden = self.weight.shape[1]
        kdim = self.weight.shape[2] * 8
        assert kdim % 16 == 0 and hidden % 16 == 0
        return _KERNEL(
            inputs=[
                act.reshape(-1), self.weight, self.scales, chosen, weights, residual,
                mx.array(hidden, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
            ],
            template=[("T", act.dtype), ("TOPK", chosen.shape[0])],
            grid=(128, hidden // 16, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(hidden,)],
            output_dtypes=[act.dtype],
        )[0]
