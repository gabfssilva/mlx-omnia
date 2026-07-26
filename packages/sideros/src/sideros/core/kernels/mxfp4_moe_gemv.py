"""The routed MXFP4 expert MLP of a one-token step in two dispatches (gpt-oss).

The gpt-oss variant of `moe_gemv`. MXFP4 is not affine: a nibble is an e2m1 code read
through a lookup, the per-group scale is an e8m0 exponent applied as `exp2(byte - 127)`,
and there is no zero point — so the affine kernel's unsigned unpack and its `xsum·bias`
term do not carry over. Each projection still adds a learned per-row bias (the
projection's, not the quantizer's), folded before the activation on the gate‖up side and
before the routing weight on the down side.

A lane reads one uint32 word — eight aligned e2m1 values, inside one quantization group —
per block. gpt-oss's contraction is 2880 = 90·32, not a multiple of the 256-value block,
so the trailing partial block is guarded rather than tiled away; the eight values still
land in one group, so the e8m0 scale is read once per word.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_HEADER = """
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

_GATE_UP_SOURCE = """
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

_DOWN_SOURCE = """
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

_GATE_UP_KERNEL = metal_kernel(
    name="mxfp4_gateup_act",
    input_names=["X", "W", "S", "Bs", "IDX", "N", "KD", "LIMIT"],
    output_names=["Y"],
    source=_GATE_UP_SOURCE,
    header=_HEADER,
)

_DOWN_KERNEL = metal_kernel(
    name="mxfp4_down_combine",
    input_names=["ACT", "W", "S", "Bs", "IDX", "WTS", "RES", "N", "KD"],
    output_names=["Y"],
    source=_DOWN_SOURCE,
    header=_HEADER,
)


def mxfp4_moe_applies(hidden: int, inner: int) -> bool:
    """A lane reads one uint32 word (eight aligned values inside one e8m0 group) per
    block, and a threadgroup writes 16 rows: the down output axis must be a multiple of
    16 and the gate‖up half-axis a multiple of eight, while both contractions have to
    cover whole 32-value groups."""
    return hidden % 32 == 0 and inner % 32 == 0


def mxfp4_gate_up_act(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    bias: mx.array,
    indices: mx.array,
    *,
    limit: float,
) -> mx.array:
    """x [hidden] through the routed row-interleaved gate‖up stacks with the clamped
    SwiGLU applied: weight [experts, 2·inner, hidden/8] uint32, e8m0 `scales`, per-row
    `bias`, `indices` [k] -> gate·sigmoid(1.702·gate)·(up+1) as [k, inner]."""
    inner = weight.shape[1] // 2
    kdim = weight.shape[2] * 8
    assert kdim % 32 == 0 and inner % 8 == 0
    # 128 threads (4 simdgroups x 4 rows = 16 output rows): more memory requests in
    # flight hides DRAM latency far better than 64. Swept in-model in the Swift era
    # against 64/192/256 — 128 wins; the isolated K=32 bandwidth disagrees, which is why
    # the model is the judge.
    return _GATE_UP_KERNEL(
        inputs=[
            x, weight, scales, bias, indices,
            mx.array(inner, dtype=mx.int32),
            mx.array(kdim, dtype=mx.int32),
            mx.array(limit, dtype=mx.float32),
        ],
        template=[("T", x.dtype)],
        grid=(128, 2 * inner // 16, indices.shape[0]),
        threadgroup=(128, 1, 1),
        output_shapes=[(indices.shape[0], inner)],
        output_dtypes=[x.dtype],
    )[0]


def mxfp4_down_combine(
    act: mx.array,
    weight: mx.array,
    scales: mx.array,
    bias: mx.array,
    indices: mx.array,
    routing: mx.array,
    residual: mx.array,
) -> mx.array:
    """act [k, inner] down-projected, per-row biased, routing-weighted, expert-summed and
    residual-added in one dispatch: weight [experts, hidden, inner/8] uint32 -> [hidden]."""
    hidden = weight.shape[1]
    kdim = weight.shape[2] * 8
    assert kdim % 32 == 0 and hidden % 16 == 0
    return _DOWN_KERNEL(
        inputs=[
            act, weight, scales, bias, indices, routing, residual,
            mx.array(hidden, dtype=mx.int32),
            mx.array(kdim, dtype=mx.int32),
        ],
        template=[("T", act.dtype), ("TOPK", indices.shape[0])],
        grid=(128, hidden // 16, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[act.dtype],
    )[0]
