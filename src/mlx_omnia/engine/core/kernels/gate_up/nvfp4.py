"""The nvfp4 gate/up kernel: the nibbles are the same e2m1 codes MXFP4 packs, but the
group is 16 values and its scale is an e4m3 byte — a small float, not an exponent. A
lane owns 16 contiguous values: exactly one quantization group, two uint32 words read
as one `uint2`, one scale byte for the pair — the tiling mlx's own `fp_qmv_fast` lands
on at group 16. The contraction is guarded rather than tiled away. No zero point, no
affine bias, no learned projection bias: the epilogue is the plain SwiGLU chain."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.gate_up.kernel import (
    Activation,
    GateUpStrategy,
    Layout,
    OrdinalRouting,
)
from mlx_omnia.engine.core.kernels.shared.nvfp4.qmoe import HEADER, applies
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint expert = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    threadgroup float NVLUT[16];
    nvfp4Lut(NVLUT, tid);

    constexpr uint block = 16 * 32;

    uint rows = 2 * (uint)N;
    uint kdim = (uint)KD;
    uint kw = kdim / 2;
    uint kg = kdim / 16u;
    uint row0 = tgy * 16 + sg * 4;
    size_t wbase = (size_t)IDX[expert] * rows;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = lane * 16, blk = 0; blk < kdim; blk += block, base += block) {
        if (base >= kdim) continue;
        float xt[16];
        for (uint j = 0; j < 16; j++) xt[j] = (float)X[base + j];
        for (uint row = 0; row < 4; row++) {
            size_t r = wbase + row0 + row;
            uint2 q = *((const device uint2*)((const device uint8_t*)W + r * kw + base / 2));
            float s = nvfp4Scale(S[r * kg + base / 16]);
            result[row] += nvfp4Dot16(q, s, xt, NVLUT);
        }
    }
    for (uint row = 0; row < 4; row++) result[row] = simd_sum(result[row]);
    if (lane == 0) {
        for (uint pair = 0; pair < 2; pair++) {
            float g = (float)(T)result[pair * 2];
            float u = (float)(T)result[pair * 2 + 1];
            float sig = (float)(T)(1.0f / (1.0f + metal::exp(-g)));
            float a = (float)(T)(g * sig);
            Y[(size_t)expert * (uint)N + tgy * 8 + sg * 2 + pair] = (T)(a * u);
        }
    }
"""

_KERNEL = metal_kernel(
    name="nvfp4_gateup_act",
    input_names=["X", "W", "S", "IDX", "N", "KD"],
    output_names=["Y"],
    source=_SOURCE,
    header=HEADER,
)


@dataclass(frozen=True)
class Nvfp4GateUp(GateUpStrategy):
    weight: mx.array
    scales: mx.array

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
        # The nvfp4 kernel carries neither clamp nor projection bias.
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "nvfp4":
            return None
        if activation != "silu" or limit is not None or bias is not None:
            return None
        if not applies(hidden, inner):
            return None
        return cls(leaf.weight, leaf.scales)

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        inner = self.weight.shape[1] // 2
        kdim = self.weight.shape[2] * 8
        assert kdim % 16 == 0 and inner % 8 == 0
        # 128 threads (4 simdgroups x 4 rows = 16 output rows), the mxfp4 sibling's
        # shape: more memory requests in flight hides DRAM latency better than 64. Not
        # swept for this format.
        return _KERNEL(
            inputs=[
                row, self.weight, self.scales, chosen,
                mx.array(inner, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
            ],
            template=[("T", row.dtype)],
            grid=(128, 2 * inner // 16, chosen.shape[0]),
            threadgroup=(128, 1, 1),
            output_shapes=[(chosen.shape[0], inner)],
            output_dtypes=[row.dtype],
        )[0]
