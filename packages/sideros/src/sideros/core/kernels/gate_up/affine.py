"""The affine gate/up kernel: silu(gate)·up computed while the dots are still in
registers (gate‖up row-interleaved at load); every op boundary of the replaced
gather_qmm chain rounds to T in place. With a clamp declared, the
clamp-before-activation order: gate capped above, up clipped both ways."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from sideros.core.kernels.gate_up.kernel import Activation
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear
from sideros.core.mxcompat import metal_kernel

HEADER = """
    template <int bits, int vpt>
    inline float qmoeDot(const device uint8_t* w, const thread float* x) {
        float d = 0.0f;
        if (bits == 4) {
            const device uint* wq = (const device uint*)w;
            for (uint p = 0; p < vpt / 8; p++) {
                uint q = wq[p];
                d += x[p*8+0] * (float)(q & 0xF) + x[p*8+1] * (float)((q >> 4) & 0xF)
                   + x[p*8+2] * (float)((q >> 8) & 0xF) + x[p*8+3] * (float)((q >> 12) & 0xF)
                   + x[p*8+4] * (float)((q >> 16) & 0xF) + x[p*8+5] * (float)((q >> 20) & 0xF)
                   + x[p*8+6] * (float)((q >> 24) & 0xF) + x[p*8+7] * (float)((q >> 28) & 0xF);
            }
        } else if (bits == 2 && vpt % 16 == 0) {
            // One hardware unpack per four bytes: the 0x03030303 mask leaves one 2-bit
            // field per byte and `unpack_unorm4x8_to_float` converts all four at once
            // (divided by 255, multiplied back at the end — folded, it is one multiply
            // per dot, not per weight). ~8% over the shift/mask/convert chain, which is
            // instruction-throughput-bound, not bandwidth-bound, at this bit width.
            const device uint* wq = (const device uint*)w;
            for (uint p = 0; p < vpt / 16; p++) {
                uint q = wq[p];
                float4 f0 = metal::unpack_unorm4x8_to_float(q & 0x03030303u);
                float4 f1 = metal::unpack_unorm4x8_to_float((q >> 2) & 0x03030303u);
                float4 f2 = metal::unpack_unorm4x8_to_float((q >> 4) & 0x03030303u);
                float4 f3 = metal::unpack_unorm4x8_to_float((q >> 6) & 0x03030303u);
                d += metal::dot(float4(x[p*16+ 0], x[p*16+ 4], x[p*16+ 8], x[p*16+12]), f0)
                   + metal::dot(float4(x[p*16+ 1], x[p*16+ 5], x[p*16+ 9], x[p*16+13]), f1)
                   + metal::dot(float4(x[p*16+ 2], x[p*16+ 6], x[p*16+10], x[p*16+14]), f2)
                   + metal::dot(float4(x[p*16+ 3], x[p*16+ 7], x[p*16+11], x[p*16+15]), f3);
            }
            d *= 255.0f;
        } else if (bits == 2) {
            const device ushort* wq = (const device ushort*)w;
            for (uint p = 0; p < vpt / 8; p++) {
                uint q = wq[p];
                for (uint j = 0; j < 8; j++) {
                    d += x[p*8+j] * (float)((q >> (2*j)) & 0x3);
                }
            }
        } else {
            constexpr uint mask = (1u << bits) - 1;
            for (uint j = 0; j < vpt; j++) {
                uint bit = j * bits;
                uint shift = bit & 7;
                uint raw = w[bit >> 3];
                if (shift + bits > 8) raw |= (uint)w[(bit >> 3) + 1] << 8;
                d += x[j] * (float)((raw >> shift) & mask);
            }
        }
        return d;
    }
"""

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint expert = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    constexpr uint values_per_thread = 16;
    constexpr uint block = values_per_thread * 32;
    constexpr uint bytes_per_thread = values_per_thread * BITS / 8;

    uint rows = 2 * (uint)N;
    uint kdim = (uint)KD;
    uint kw_bytes = kdim * BITS / 8;
    uint kg = kdim / (uint)GSIZE;
    uint lanes_per_group = (uint)GSIZE / values_per_thread;

    uint row0 = tgy * 8 + sg * 4;
    size_t wbase = (size_t)IDX[expert] * rows;

    const device uint8_t* ws = (const device uint8_t*)W
        + (wbase + row0) * kw_bytes + lane * bytes_per_thread;
    const device T* sl = S + (wbase + row0) * kg + lane / lanes_per_group;
    const device T* bl = Bs + (wbase + row0) * kg + lane / lanes_per_group;
    const device T* xv = X + lane * values_per_thread;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < kdim; k += block) {
        float xt[values_per_thread];
        float xsum = 0.0f;
        for (uint j = 0; j < values_per_thread; j++) {
            xt[j] = (float)xv[j];
            xsum += xt[j];
        }
        for (uint row = 0; row < 4; row++) {
            float s = (float)sl[row * kg];
            float b = (float)bl[row * kg];
            float d = qmoeDot<BITS, values_per_thread>(ws + row * kw_bytes, xt);
            result[row] += d * s + xsum * b;
        }
        ws += block * BITS / 8;
        sl += block / (uint)GSIZE;
        bl += block / (uint)GSIZE;
        xv += block;
    }
    for (uint row = 0; row < 4; row++) {
        result[row] = simd_sum(result[row]);
    }
    if (lane == 0) {
        float limit = (float)LIM;
        for (uint pair = 0; pair < 2; pair++) {
            float g = (float)(T)result[pair * 2];
            float u = (float)(T)result[pair * 2 + 1];
            if (metal::isfinite(limit)) {
                g = metal::min(g, limit);
                u = metal::min(metal::max(u, -limit), limit);
            }
            float sg_ = (float)(T)(1.0f / (1.0f + metal::exp(-g)));
            float a = (float)(T)(g * sg_);
            Y[(size_t)expert * (uint)N + tgy * 4 + sg * 2 + pair] = (T)(a * u);
        }
    }
"""

_KERNEL = metal_kernel(
    name="moe_gateup_act",
    input_names=["X", "W", "S", "Bs", "IDX", "LIM", "N", "KD", "GSIZE"],
    output_names=["Y"],
    source=_SOURCE,
    header=HEADER,
)


def applies(hidden: int, inner: int, group: int) -> bool:
    """16 values/lane over the hidden contraction, a lane's values inside one
    quantization group, and the output rows the grid divides by. Bit width never
    disqualifies."""
    return hidden % 512 == 0 and group % 16 == 0 and hidden % group == 0 and inner % 4 == 0


@dataclass(frozen=True)
class AffineGateUp:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int
    limit: float | None

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
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "affine":
            return None
        if activation != "silu" or bias is not None:
            return None
        if leaf.biases is None or not applies(hidden, inner, leaf.group_size):
            return None
        return cls(leaf.weight, leaf.scales, leaf.biases, leaf.group_size, leaf.bits, limit)

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        inner = self.weight.shape[1] // 2
        kdim = self.weight.shape[2] * 32 // self.bits
        assert self.bits <= 8 and kdim % 512 == 0 and self.group_size % 16 == 0 and inner % 4 == 0
        return _KERNEL(
            inputs=[
                row, self.weight, self.scales, self.biases, chosen,
                mx.array(float("inf") if self.limit is None else self.limit, dtype=mx.float32),
                mx.array(inner, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
                mx.array(self.group_size, dtype=mx.int32),
            ],
            template=[("T", row.dtype), ("BITS", self.bits)],
            grid=(64, 2 * inner // 8, chosen.shape[0]),
            threadgroup=(64, 1, 1),
            output_shapes=[(chosen.shape[0], inner)],
            output_dtypes=[row.dtype],
        )[0]
