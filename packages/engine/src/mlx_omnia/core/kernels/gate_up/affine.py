"""The affine gate/up kernel: silu(gate)·up computed while the dots are still in
registers (gate‖up row-interleaved at load); every op boundary of the replaced
gather_qmm chain rounds to T in place. With a clamp declared, the
clamp-before-activation order: gate capped above, up clipped both ways.

The last slot serves a declared shared expert from its own leaf, on its own bit width and
group size: a stack holds one format and a per-leaf plan gives that expert another."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.gate_up.kernel import Activation, Layout, OrdinalRouting
from mlx_omnia.core.kernels.shared.affine import HEADER
from mlx_omnia.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.core.mxcompat import metal_kernel

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint expert = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    constexpr uint values_per_thread = 16;
    constexpr uint block = values_per_thread * 32;

    // The last slot is the shared expert's own leaf when one is declared: its own
    // buffers, its own width and group. The branch is uniform over the threadgroup
    // (the slot is the grid's z), and folds away entirely without the declaration.
    bool last = SHARED && expert == (uint)TOPK - 1;
    uint bits = last ? (uint)SBITS : (uint)BITS;
    uint group = last ? (uint)SGSIZE : (uint)GSIZE;
    uint bytes_per_thread = values_per_thread * bits / 8;

    uint rows = 2 * (uint)N;
    uint kdim = (uint)KD;
    uint kw_bytes = kdim * bits / 8;
    uint kg = kdim / group;
    uint lanes_per_group = group / values_per_thread;

    uint row0 = tgy * 8 + sg * 4;
    size_t wbase = last ? 0 : (size_t)IDX[expert] * rows;

    const device uint8_t* ws = (const device uint8_t*)(last ? SW : W)
        + (wbase + row0) * kw_bytes + lane * bytes_per_thread;
    const device T* sl = (last ? SS : S) + (wbase + row0) * kg + lane / lanes_per_group;
    const device T* bl = (last ? SB : Bs) + (wbase + row0) * kg + lane / lanes_per_group;
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
            float d = last
                ? qmoeDot<SBITS, values_per_thread>(ws + row * kw_bytes, xt)
                : qmoeDot<BITS, values_per_thread>(ws + row * kw_bytes, xt);
            result[row] += d * s + xsum * b;
        }
        ws += block * bits / 8;
        sl += block / group;
        bl += block / group;
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
    input_names=["X", "W", "S", "Bs", "IDX", "LIM", "N", "KD", "GSIZE", "SW", "SS", "SB", "SGSIZE"],
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
    shared: tuple[mx.array, mx.array, mx.array, int, int] | None

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
        if layout != "interleaved" or routing is not None:
            return None
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "affine":
            return None
        if activation != "silu" or bias is not None:
            return None
        if leaf.biases is None or not applies(hidden, inner, leaf.group_size):
            return None
        spare: tuple[mx.array, mx.array, mx.array, int, int] | None = None
        if shared is not None:
            # The spare slot decodes on its own width and group — what it cannot be is a
            # format this kernel has no decoder for, or a geometry the tiling misses.
            if not isinstance(shared, nn.QuantizedLinear) or shared.mode != "affine":
                return None
            if not isinstance(shared.biases, mx.array) or shared.bits > 8:
                return None
            if "bias" in shared or not applies(hidden, inner, shared.group_size):
                return None
            spare = (
                shared.weight, shared.scales, shared.biases, shared.group_size, shared.bits
            )
        return cls(
            leaf.weight, leaf.scales, leaf.biases, leaf.group_size, leaf.bits, limit, spare
        )

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        inner = self.weight.shape[1] // 2
        kdim = self.weight.shape[2] * 32 // self.bits
        assert self.bits <= 8 and kdim % 512 == 0 and self.group_size % 16 == 0 and inner % 4 == 0
        spare = self.shared
        weight, scales, biases, group, bits = (
            spare
            if spare is not None
            else (self.weight, self.scales, self.biases, self.group_size, self.bits)
        )
        return _KERNEL(
            inputs=[
                row, self.weight, self.scales, self.biases, chosen,
                mx.array(float("inf") if self.limit is None else self.limit, dtype=mx.float32),
                mx.array(inner, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
                mx.array(self.group_size, dtype=mx.int32),
                weight, scales, biases,
                mx.array(group, dtype=mx.int32),
            ],
            template=[
                ("T", row.dtype), ("BITS", self.bits), ("SBITS", bits),
                ("TOPK", chosen.shape[0]), ("SHARED", 1 if spare is not None else 0),
            ],
            grid=(64, 2 * inner // 8, chosen.shape[0]),
            threadgroup=(64, 1, 1),
            output_shapes=[(chosen.shape[0], inner)],
            output_dtypes=[row.dtype],
        )[0]
