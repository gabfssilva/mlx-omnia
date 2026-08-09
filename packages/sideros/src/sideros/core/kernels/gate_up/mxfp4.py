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

from sideros.core.kernels.gate_up.kernel import Activation
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear
from sideros.core.mxcompat import metal_kernel

HEADER = """
    // The signed e2m1 lookup: the full nibble (sign bit included) indexes the value
    // directly, so the dot loop is a table load and an fma — no mask, no sign branch.
    // The table lives in threadgroup memory, not `constant`: each lane looks up a
    // different nibble and `constant` memory is broadcast-optimized, so divergent reads
    // serialize it (measured 345 vs 608 GB/s in the Swift era). Seeded once per
    // threadgroup.
    inline void mxfp4Lut(threadgroup float* L, uint tid) {
        if (tid < 16) {
            const float v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            L[tid] = v[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Eight e2m1 nibbles from one uint32 word, scaled by their shared group's e8m0
    // exponent — which is a float's exponent field, so 2^(byte-127) is a shift, not a
    // transcendental.
    inline float mxfp4Dot8(uint q, float s, const thread float* x,
                           threadgroup const float* L) {
        float d = 0.0f;
        for (uint n = 0; n < 8; n++) d += x[n] * L[(q >> (n * 4)) & 0xF];
        return d * s;
    }
"""

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


def applies(hidden: int, inner: int) -> bool:
    """A lane reads one uint32 word (eight aligned values inside one e8m0 group) per
    block, and both contractions have to cover whole 32-value groups."""
    return hidden % 32 == 0 and inner % 32 == 0


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
    ) -> Self | None:
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
