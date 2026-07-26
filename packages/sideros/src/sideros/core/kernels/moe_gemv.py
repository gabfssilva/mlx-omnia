"""The routed quantized expert MLP of a one-token step in two dispatches.

gather_qmm plus the elementwise chain needs five. The first dispatch computes
silu(gate)·up while the dots are still in registers (gate‖up row-interleaved at
load); the second folds routing weights, expert sum and residual into the down
projection. Every op boundary of the replaced chain rounds to T in place.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_HEADER = """
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

_GATE_UP_SOURCE = """
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
        for (uint pair = 0; pair < 2; pair++) {
            float g = (float)(T)result[pair * 2];
            float u = (float)(T)result[pair * 2 + 1];
            float sg_ = (float)(T)(1.0f / (1.0f + metal::exp(-g)));
            float a = (float)(T)(g * sg_);
            Y[(size_t)expert * (uint)N + tgy * 4 + sg * 2 + pair] = (T)(a * u);
        }
    }
"""

_DOWN_SOURCE = """
    uint tgy = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    constexpr uint values_per_thread = 8;
    constexpr uint block = values_per_thread * 32;
    constexpr uint bytes_per_thread = values_per_thread * BITS / 8;

    uint kdim = (uint)KD;
    uint kg = kdim / (uint)GSIZE;
    uint kw_bytes = kdim * BITS / 8;
    uint lanes_per_group = (uint)GSIZE / values_per_thread;

    uint row0 = tgy * 8 + sg * 4;
    uint rows = (uint)N;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint e = 0; e < (uint)TOPK; e++) {
        bool last = SHARED && e == (uint)TOPK - 1;
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
                float s = (float)sl[row * kg + k / (uint)GSIZE];
                float b = (float)bl[row * kg + k / (uint)GSIZE];
                float d = qmoeDot<BITS, values_per_thread>(
                    ws + row * kw_bytes + k * BITS / 8, xt);
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

_GATE_UP_KERNEL = metal_kernel(
    name="moe_gateup_act",
    input_names=["X", "W", "S", "Bs", "IDX", "N", "KD", "GSIZE"],
    output_names=["Y"],
    source=_GATE_UP_SOURCE,
    header=_HEADER,
)

_DOWN_KERNEL = metal_kernel(
    name="moe_down_combine",
    input_names=["ACT", "W", "S", "Bs", "IDX", "WTS", "RES", "SW", "SS", "SB", "N", "KD", "GSIZE"],
    output_names=["Y"],
    source=_DOWN_SOURCE,
    header=_HEADER,
)


def moe_gemv_applies(hidden: int, inner: int, gate_up_group: int, down_group: int) -> bool:
    """The reduction has to tile: 16 values/lane gate-up side, 8 down side, and a
    lane's values inside one quantization group. Bit width never disqualifies."""

    def tiles(kdim: int, group: int, per_lane: int) -> bool:
        return kdim % (per_lane * 32) == 0 and group % per_lane == 0 and kdim % group == 0

    return (
        tiles(hidden, gate_up_group, 16)
        and tiles(inner, down_group, 8)
        and inner % 4 == 0
        and hidden % 8 == 0
    )


def moe_gate_up_act(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    """x [hidden] through the routed row-interleaved gate‖up stacks -> silu(gate)·up [k, inner]."""
    inner = weight.shape[1] // 2
    kdim = weight.shape[2] * 32 // bits
    assert bits <= 8 and kdim % 512 == 0 and group_size % 16 == 0 and inner % 4 == 0
    return _GATE_UP_KERNEL(
        inputs=[
            x, weight, scales, biases, indices,
            mx.array(inner, dtype=mx.int32),
            mx.array(kdim, dtype=mx.int32),
            mx.array(group_size, dtype=mx.int32),
        ],
        template=[("T", x.dtype), ("BITS", bits)],
        grid=(64, 2 * inner // 8, indices.shape[0]),
        threadgroup=(64, 1, 1),
        output_shapes=[(indices.shape[0], inner)],
        output_dtypes=[x.dtype],
    )[0]


def moe_down_combine(
    act: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    indices: mx.array,
    routing: mx.array,
    residual: mx.array,
    *,
    group_size: int,
    bits: int,
    shared: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    """act [k, inner] down-projected, routing-weighted, expert-summed,
    residual-added -> [hidden]."""
    hidden = weight.shape[1]
    kdim = weight.shape[2] * 32 // bits
    assert bits <= 8 and kdim % 256 == 0 and group_size % 8 == 0 and hidden % 8 == 0
    spare = shared if shared is not None else (weight, scales, biases)
    return _DOWN_KERNEL(
        inputs=[
            act, weight, scales, biases, indices, routing, residual,
            spare[0], spare[1], spare[2],
            mx.array(hidden, dtype=mx.int32),
            mx.array(kdim, dtype=mx.int32),
            mx.array(group_size, dtype=mx.int32),
        ],
        template=[
            ("T", act.dtype), ("BITS", bits), ("TOPK", indices.shape[0]),
            ("SHARED", 1 if shared is not None else 0),
        ],
        grid=(64, hidden // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[act.dtype],
    )[0]
