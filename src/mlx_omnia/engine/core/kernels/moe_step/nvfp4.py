"""The nvfp4 T=1 sparse step: two dispatches for the whole routed stack.

The op chain runs one `gather_qmm` per projection plus the elementwise middle and the
combine; here the k experts ride one grid per projection. The e2m1/e4m3 decode and the
lane-owns-one-group tiling are `shared.nvfp4.qmoe`'s, the same bytes and the same fp32
accumulation order as mlx's `fp_qmv_fast`, so each projection's rows match the stock
kernel's bit-for-bit. The epilogues carry the op chain's rounding: squared ReLU squares the
T-rounded row; the combine multiplies the T-rounded row by the fp32 router weight,
accumulates in fp32 expert order, casts once, and adds the shared row in T.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.moe_step.default import SharedPair
from mlx_omnia.engine.core.kernels.shared.nvfp4.qmoe import HEADER, applies
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_UP_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint expert = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    threadgroup float NVLUT[16];
    nvfp4Lut(NVLUT, tid);

    constexpr uint block = 16 * 32;

    bool unrouted = SHARED && expert == (uint)TOPK;
    uint rows = unrouted ? (uint)N2 : (uint)N;
    uint kdim = (uint)KD;
    uint kw = kdim / 2;
    uint kg = kdim / 16u;
    uint row0 = tgy * 16 + sg * 4;
    if (row0 >= rows) return;
    const device uint8_t* wbytes = unrouted
        ? (const device uint8_t*)SUW
        : (const device uint8_t*)W + (size_t)IDX[expert] * rows * kw;
    const device uint8_t* sbytes = unrouted
        ? (const device uint8_t*)SUS
        : (const device uint8_t*)S + (size_t)IDX[expert] * rows * kg;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = lane * 16, blk = 0; blk < kdim; blk += block, base += block) {
        if (base >= kdim) continue;
        float xt[16];
        for (uint j = 0; j < 16; j++) xt[j] = (float)X[base + j];
        for (uint row = 0; row < 4; row++) {
            size_t r = row0 + row;
            uint2 q = *((const device uint2*)(wbytes + r * kw + base / 2));
            float s = nvfp4Scale(sbytes[r * kg + base / 16]);
            result[row] += nvfp4Dot16(q, s, xt, NVLUT);
        }
    }
    for (uint row = 0; row < 4; row++) result[row] = simd_sum(result[row]);
    if (lane == 0) {
        for (uint row = 0; row < 4; row++) {
            uint i = row0 + row;
            if (i < rows) {
                float t = (float)(T)result[row];
                float a = metal::max(t, 0.0f);
                T v = (T)((T)a * (T)a);
                if (unrouted) { Y2[i] = v; } else { Y[(size_t)expert * (uint)N + i] = v; }
            }
        }
    }
"""

_DOWN_SOURCE = """
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
        const device T* xe = ACT + (size_t)e * kdim;
        float wt = WTS[e];
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
                acc[row] += (float)(T)t * wt;
            }
        }
    }
    float sh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (SHARED) {
        uint kdim2 = (uint)KD2;
        uint kw2 = kdim2 / 2;
        uint kg2 = kdim2 / 16u;
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint base = lane * 16, blk = 0; blk < kdim2; blk += block, base += block) {
            if (base >= kdim2) continue;
            float xt[16];
            for (uint j = 0; j < 16; j++) xt[j] = (float)ACT2[base + j];
            for (uint row = 0; row < 4; row++) {
                size_t r = row0 + row;
                uint2 q = *((const device uint2*)((const device uint8_t*)SDW + r * kw2 + base / 2));
                float s = nvfp4Scale(SDS[r * kg2 + base / 16]);
                res[row] += nvfp4Dot16(q, s, xt, NVLUT);
            }
        }
        for (uint row = 0; row < 4; row++) {
            float t = simd_sum(res[row]);
            if (lane == 0) sh[row] = (float)(T)t;
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < 4; row++) {
            uint i = row0 + row;
            if (i < rows) {
                float base_ = SHARED ? sh[row] : 0.0f;
                Y[i] = (T)((float)(T)acc[row] + base_);
            }
        }
    }
"""

_UP = metal_kernel(
    name="nvfp4_up_squared_relu",
    input_names=["X", "W", "S", "IDX", "SUW", "SUS", "N", "KD", "N2"],
    output_names=["Y", "Y2"],
    source=_UP_SOURCE,
    header=HEADER,
)

_DOWN = metal_kernel(
    name="nvfp4_down_combine_squared_relu",
    input_names=["ACT", "W", "S", "IDX", "WTS", "ACT2", "SDW", "SDS", "N", "KD", "KD2"],
    output_names=["Y"],
    source=_DOWN_SOURCE,
    header=HEADER,
)



_UP_ROWS_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint pair = threadgroup_position_in_grid.z;
    uint row_t = pair / (uint)TOPK;
    uint expert = pair % (uint)TOPK;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    threadgroup float NVLUT[16];
    nvfp4Lut(NVLUT, tid);

    constexpr uint block = 16 * 32;

    uint rows = (uint)N;
    uint kdim = (uint)KD;
    uint kw = kdim / 2;
    uint kg = kdim / 16u;
    uint row0 = tgy * 16 + sg * 4;
    if (row0 >= rows) return;
    const device T* x = X + (size_t)row_t * kdim;
    size_t wbase = (size_t)IDX[pair] * rows;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = lane * 16, blk = 0; blk < kdim; blk += block, base += block) {
        if (base >= kdim) continue;
        float xt[16];
        for (uint j = 0; j < 16; j++) xt[j] = (float)x[base + j];
        for (uint row = 0; row < 4; row++) {
            size_t r = wbase + row0 + row;
            uint2 q = *((const device uint2*)((const device uint8_t*)W + r * kw + base / 2));
            float s = nvfp4Scale(S[r * kg + base / 16]);
            result[row] += nvfp4Dot16(q, s, xt, NVLUT);
        }
    }
    for (uint row = 0; row < 4; row++) result[row] = simd_sum(result[row]);
    if (lane == 0) {
        for (uint row = 0; row < 4; row++) {
            uint i = row0 + row;
            if (i < rows) {
                float t = (float)(T)result[row];
                float a = metal::max(t, 0.0f);
                Y[(size_t)pair * rows + i] = (T)((T)a * (T)a);
            }
        }
    }
"""

_DOWN_ROWS_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint row_t = threadgroup_position_in_grid.z;
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
    if (row0 >= rows) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint e = 0; e < (uint)TOPK; e++) {
        uint pair = row_t * (uint)TOPK + e;
        size_t wbase = (size_t)IDX[pair] * rows;
        const device T* xe = ACT + (size_t)pair * kdim;
        float wt = WTS[pair];
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
                acc[row] += (float)(T)t * wt;
            }
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < 4; row++) {
            uint i = row0 + row;
            if (i < rows) {
                Y[(size_t)row_t * rows + i] = (T)acc[row];
            }
        }
    }
"""

_UP_ROWS = metal_kernel(
    name="nvfp4_up_squared_relu_rows",
    input_names=["X", "W", "S", "IDX", "N", "KD"],
    output_names=["Y"],
    source=_UP_ROWS_SOURCE,
    header=HEADER,
)

_DOWN_ROWS = metal_kernel(
    name="nvfp4_down_combine_rows",
    input_names=["ACT", "W", "S", "IDX", "WTS", "N", "KD"],
    output_names=["Y"],
    source=_DOWN_ROWS_SOURCE,
    header=HEADER,
)


def _ceil16(rows: int) -> int:
    return (rows + 15) // 16


def _nvfp4_linear(leaf: object) -> bool:
    import mlx.nn as nn

    return (
        isinstance(leaf, nn.QuantizedLinear)
        and leaf.mode == "nvfp4"
        and "bias" not in leaf
    )


@dataclass(frozen=True)
class Nvfp4MoeStep:
    fc1: QuantizedSwitchLinear
    fc2: QuantizedSwitchLinear
    hidden: int
    inner: int
    shared_up: "mx.array | None"
    shared_up_scales: "mx.array | None"
    shared_down: "mx.array | None"
    shared_down_scales: "mx.array | None"
    shared_inner: int

    @classmethod
    def build(
        cls,
        *,
        fc1: SwitchLinear | QuantizedSwitchLinear,
        fc2: SwitchLinear | QuantizedSwitchLinear,
        hidden: int,
        inner: int,
        shared: SharedPair | None = None,
    ) -> Self | None:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
        if not isinstance(fc1, QuantizedSwitchLinear) or fc1.mode != "nvfp4":
            return None
        if not isinstance(fc2, QuantizedSwitchLinear) or fc2.mode != "nvfp4":
            return None
        if not (applies(hidden, inner) and applies(inner, hidden)):
            return None
        if shared is None:
            return cls(fc1, fc2, hidden, inner, None, None, None, None, 0)
        up, down = shared
        if not (_nvfp4_linear(up) and _nvfp4_linear(down)):
            return None
        import mlx.nn as nn

        assert isinstance(up, nn.QuantizedLinear) and isinstance(down, nn.QuantizedLinear)
        shared_inner = up.weight.shape[0]
        if not (applies(hidden, shared_inner) and applies(shared_inner, hidden)):
            return None
        return cls(
            fc1, fc2, hidden, inner,
            up.weight, up.scales, down.weight, down.scales, shared_inner,
        )

    def __call__(self, x: mx.array, chosen: mx.array, weights: mx.array) -> mx.array:
        k = chosen.shape[0]
        fused_shared = self.shared_up is not None
        # Placeholders keep the unused buffers in device address space; a SHARED=0
        # kernel never reads them.
        up, up2 = _UP(
            inputs=[
                x, self.fc1.weight, self.fc1.scales, chosen,
                self.shared_up if self.shared_up is not None else self.fc1.weight,
                self.shared_up_scales
                if self.shared_up_scales is not None
                else self.fc1.scales,
                mx.array(self.inner, dtype=mx.int32),
                mx.array(self.hidden, dtype=mx.int32),
                mx.array(self.shared_inner, dtype=mx.int32),
            ],
            template=[("T", x.dtype), ("TOPK", k), ("SHARED", fused_shared)],
            grid=(
                128,
                _ceil16(max(self.inner, self.shared_inner)),
                k + (1 if fused_shared else 0),
            ),
            threadgroup=(128, 1, 1),
            output_shapes=[(k, self.inner), (max(self.shared_inner, 1),)],
            output_dtypes=[x.dtype, x.dtype],
        )
        return _DOWN(
            inputs=[
                up, self.fc2.weight, self.fc2.scales, chosen, weights,
                up2,
                self.shared_down if self.shared_down is not None else self.fc2.weight,
                self.shared_down_scales
                if self.shared_down_scales is not None
                else self.fc2.scales,
                mx.array(self.hidden, dtype=mx.int32),
                mx.array(self.inner, dtype=mx.int32),
                mx.array(self.shared_inner, dtype=mx.int32),
            ],
            template=[("T", x.dtype), ("TOPK", k), ("SHARED", fused_shared)],
            grid=(128, _ceil16(self.hidden), 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(self.hidden,)],
            output_dtypes=[x.dtype],
        )[0]

    def rows(
        self, x: mx.array, chosen: mx.array, weights: mx.array
    ) -> mx.array:
        """The routed stack over T token rows in the same two dispatches: each
        (row, expert) pair rides the grid, and the combine closes per row without the
        shared expert — the caller adds it, the way the T-row op chain does. Same
        arithmetic as the single-row path, row for row."""
        tokens, k = chosen.shape
        flat_idx = chosen.reshape(-1)
        up = _UP_ROWS(
            inputs=[
                x, self.fc1.weight, self.fc1.scales, flat_idx,
                mx.array(self.inner, dtype=mx.int32),
                mx.array(self.hidden, dtype=mx.int32),
            ],
            template=[("T", x.dtype), ("TOPK", k)],
            grid=(128, _ceil16(self.inner), tokens * k),
            threadgroup=(128, 1, 1),
            output_shapes=[(tokens * k, self.inner)],
            output_dtypes=[x.dtype],
        )[0]
        return _DOWN_ROWS(
            inputs=[
                up, self.fc2.weight, self.fc2.scales, flat_idx,
                weights.reshape(-1),
                mx.array(self.hidden, dtype=mx.int32),
                mx.array(self.inner, dtype=mx.int32),
            ],
            template=[("T", x.dtype), ("TOPK", k)],
            grid=(128, _ceil16(self.hidden), tokens),
            threadgroup=(128, 1, 1),
            output_shapes=[(tokens, self.hidden)],
            output_dtypes=[x.dtype],
        )[0]
