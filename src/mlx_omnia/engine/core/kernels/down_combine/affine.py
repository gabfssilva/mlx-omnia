"""The affine down/combine kernel. The dequant helper (`qmoeDot`) is shared with the
gate-up half, from `shared.affine` — same bytes on both sides of the step. The
spare slot serves a shared expert on its own bit width and group size: a stack holds one
format and a per-leaf plan gives that expert another."""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.down_combine.kernel import DownCombineStrategy, Layout
from mlx_omnia.engine.core.kernels.shared.affine import HEADER
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    constexpr uint values_per_thread = VPT;
    constexpr uint block = values_per_thread * 32;

    uint kdim = (uint)KD;
    uint row0 = tgy * 8 + sg * 4;
    uint rows = (uint)N;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint e = 0; e < (uint)TOPK; e++) {
        // The shared slot reads its own leaf on its own width and group; without the
        // declaration `last` is a compile-time false and the whole branch folds away.
        bool last = SHARED && e == (uint)TOPK - 1;
        uint bits = last ? (uint)SBITS : (uint)BITS;
        uint group = last ? (uint)SGSIZE : (uint)GSIZE;
        uint bytes_per_thread = values_per_thread * bits / 8;
        uint kg = kdim / group;
        uint kw_bytes = kdim * bits / 8;
        uint lanes_per_group = group / values_per_thread;
        size_t base = last ? (size_t)row0 : (size_t)IDX[e] * rows + row0;
        const device uint8_t* ws = (const device uint8_t*)(last ? SW : W)
            + base * kw_bytes + lane * bytes_per_thread;
        const device T* sl = (last ? SS : S) + base * kg + lane / lanes_per_group;
        const device T* bl = (last ? SB : Bs) + base * kg + lane / lanes_per_group;
        const device T* xe = ACT + e * kdim;
        float wt = (float)WTS[e];
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint k = 0; k < kdim; k += block) {
            float xt[values_per_thread];
            float xs_ = 0.0f;
            for (uint j = 0; j < values_per_thread; j++) {
                xt[j] = (float)xe[k + lane * values_per_thread + j];
                xs_ += xt[j];
            }
            for (uint row = 0; row < 4; row++) {
                float s = (float)sl[row * kg + k / group];
                float b = (float)bl[row * kg + k / group];
                float d = last
                    ? qmoeDot<SBITS, values_per_thread>(
                        ws + row * kw_bytes + k * bits / 8, xt)
                    : qmoeDot<BITS, values_per_thread>(
                        ws + row * kw_bytes + k * bits / 8, xt);
                res[row] += d * s + xs_ * b;
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
    name="moe_down_combine",
    input_names=[
        "ACT", "W", "S", "Bs", "IDX", "WTS", "RES", "SW", "SS", "SB", "N", "KD", "GSIZE", "SGSIZE"
    ],
    output_names=["Y"],
    source=_SOURCE,
    header=HEADER,
)


def applies(hidden: int, inner: int, group: int) -> bool:
    """8 values/lane over the inner contraction, a lane's values inside one
    quantization group, and the output rows the grid divides by."""
    return inner % 256 == 0 and group % 8 == 0 and inner % group == 0 and hidden % 8 == 0


@dataclass(frozen=True)
class AffineDownCombine(DownCombineStrategy):
    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int
    shared: tuple[mx.array, mx.array, mx.array, int, int] | None

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
        if not isinstance(leaf, QuantizedSwitchLinear) or leaf.mode != "affine":
            return None
        if bias is not None or leaf.biases is None:
            return None
        if not applies(hidden, inner, leaf.group_size):
            return None
        packed: tuple[mx.array, mx.array, mx.array, int, int] | None = None
        if shared is not None:
            # The spare slot decodes on its own width and group; what it cannot be is a
            # mode this kernel has no decoder for, or a geometry the tiling misses.
            if not isinstance(shared, nn.QuantizedLinear) or shared.mode != "affine":
                return None
            if not isinstance(shared.biases, mx.array) or shared.bits > 8:
                return None
            if "bias" in shared or not applies(hidden, inner, shared.group_size):
                return None
            packed = (
                shared.weight, shared.scales, shared.biases, shared.group_size, shared.bits
            )
        return cls(leaf.weight, leaf.scales, leaf.biases, leaf.group_size, leaf.bits, packed)

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array:
        hidden = self.weight.shape[1]
        kdim = self.weight.shape[2] * 32 // self.bits
        assert self.bits <= 8 and kdim % 256 == 0 and self.group_size % 8 == 0 and hidden % 8 == 0
        spare = self.shared or (self.weight, self.scales, self.biases, self.group_size, self.bits)
        # 16 values per lane rides the word-at-a-time 2-bit unpack; every other width
        # keeps the 8-value tiling the applies() contract was written for. The spare
        # slot's group has to hold a lane's values too, whatever its width decodes as.
        vpt = (
            16
            if self.bits == 2
            and kdim % 512 == 0
            and self.group_size % 16 == 0
            and spare[3] % 16 == 0
            else 8
        )
        return _KERNEL(
            inputs=[
                act.reshape(-1), self.weight, self.scales, self.biases, chosen, weights,
                residual,
                spare[0], spare[1], spare[2],
                mx.array(hidden, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
                mx.array(self.group_size, dtype=mx.int32),
                mx.array(spare[3], dtype=mx.int32),
            ],
            template=[
                ("T", act.dtype), ("BITS", self.bits), ("SBITS", spare[4]),
                ("TOPK", chosen.shape[0]),
                ("SHARED", 1 if self.shared is not None else 0), ("VPT", vpt),
            ],
            grid=(64, hidden // 8, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(hidden,)],
            output_dtypes=[act.dtype],
        )[0]
