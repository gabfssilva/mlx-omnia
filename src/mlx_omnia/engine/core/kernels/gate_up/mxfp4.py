"""The mxfp4 gate/up kernel: a nibble is an e2m1 code read through a lookup, the
per-group scale is an e8m0 exponent applied as `exp2(byte - 127)`, no zero point.
Each projection adds a learned per-row bias (the projection's, not the quantizer's),
folded before the activation. A lane reads one uint32 word — eight aligned values
inside one group — per block; a contraction that is not a multiple of the 256-value
block is guarded rather than tiled away. The kernel bakes the swiglu_oai activation
(`gate·sigmoid(1.702·gate)·(up+1)`) and its mandatory clamp."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.gate_up.kernel import Activation, Layout, OrdinalRouting
from mlx_omnia.engine.core.kernels.shared.mxfp4 import HEADER, applies
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint expert = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    threadgroup float MXLUT[16];
    mxfp4Lut(MXLUT, tid);

    constexpr uint block = 8 * 32;

    uint rows = 2 * (uint)N;
    uint kdim = (uint)KD;
    uint kw = kdim / 2;
    uint kg = kdim / 32u;
    uint row0 = tgy * 16 + sg * 4;
    size_t wbase = (size_t)IDX[expert] * rows;
    const device T* bl = Bs + wbase + row0;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = lane * 8, blk = 0; blk < kdim; blk += block, base += block) {
        if (base >= kdim) continue;
        float xt[8];
        for (uint j = 0; j < 8; j++) xt[j] = (float)X[base + j];
        for (uint row = 0; row < 4; row++) {
            uint q = *((const device uint*)((const device uint8_t*)W
                + (wbase + row0 + row) * kw + base / 2));
            float s = as_type<float>((uint)S[(wbase + row0 + row) * kg + base / 32] << 23);
            result[row] += mxfp4Dot8(q, s, xt, MXLUT);
        }
    }
    for (uint row = 0; row < 4; row++) result[row] = simd_sum(result[row]);
    if (lane == 0) {
        for (uint pair = 0; pair < 2; pair++) {
            float g = (float)(T)(result[pair * 2] + (float)bl[pair * 2]);
            float u = (float)(T)(result[pair * 2 + 1] + (float)bl[pair * 2 + 1]);
            g = (float)(T)metal::min(g, (float)LIMIT);
            u = (float)(T)metal::clamp(u, -(float)LIMIT, (float)LIMIT);
            float sig = (float)(T)(1.0f / (1.0f + metal::exp(-1.702f * g)));
            float a = (float)(T)(g * sig);
            Y[(size_t)expert * (uint)N + tgy * 8 + sg * 2 + pair] =
                (T)((float)(T)(a * (float)(T)(u + 1.0f)));
        }
    }
"""

_KERNEL = metal_kernel(
    name="mxfp4_gateup_act",
    input_names=["X", "W", "S", "Bs", "IDX", "N", "KD", "LIMIT"],
    output_names=["Y"],
    source=_SOURCE,
    header=HEADER,
)


@dataclass(frozen=True)
class Mxfp4GateUp:
    weight: mx.array
    scales: mx.array
    bias: mx.array
    limit: float

    @classmethod
    def build(
        cls,
        leaf: SwitchLinear | QuantizedSwitchLinear,
        *,
        hidden: int,
        inner: int,
        activation: Activation,
        limit: float | None,
        bias: mx.array | None,
        layout: Layout,
        routing: OrdinalRouting | None,
        shared: nn.Linear | nn.QuantizedLinear | None,
    ) -> Self | None:
        # The spare slot is the affine kernel's; every other format runs the routed
        # stack alone, so a shared expert beside it has nowhere to go.
        if shared is not None:
            return None
        if layout != "interleaved" or routing is not None:
            return None
        # The kernel bakes the swiglu_oai activation, its mandatory clamp and the
        # per-row projection bias; anything less declared has no kernel here.
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "mxfp4":
            return None
        if activation != "swiglu_oai" or limit is None or bias is None:
            return None
        if not applies(hidden, inner):
            return None
        return cls(leaf.weight, leaf.scales, bias, limit)

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        inner = self.weight.shape[1] // 2
        kdim = self.weight.shape[2] * 8
        assert kdim % 32 == 0 and inner % 8 == 0
        # 128 threads (4 simdgroups x 4 rows = 16 output rows): more memory requests in
        # flight hides DRAM latency far better than 64. Swept in-model in the Swift era
        # against 64/192/256 — 128 wins; the isolated K=32 bandwidth disagrees, which is
        # why the model is the judge.
        return _KERNEL(
            inputs=[
                row, self.weight, self.scales, self.bias, chosen,
                mx.array(inner, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
                mx.array(self.limit, dtype=mx.float32),
            ],
            template=[("T", row.dtype)],
            grid=(128, 2 * inner // 16, chosen.shape[0]),
            threadgroup=(128, 1, 1),
            output_shapes=[(chosen.shape[0], inner)],
            output_dtypes=[row.dtype],
        )[0]
