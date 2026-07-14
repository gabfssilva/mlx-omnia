import MLX

/// The unpack, in one place, templated on the width. Every MLX affine quantization is a
/// contiguous little-endian bit stream over uint32 words — 2, 3, 4, 5, 6 and 8 bits alike
/// — so a value spans at most two bytes, and the second is read only when it straddles.
/// The width never disqualifies the kernel; only a shape that fails to tile it can.
///
/// `bits == 4` keeps a loop of its own: two packed words, no byte traffic. It is the path
/// the 30B is measured on, and it stays here textually so its instruction schedule — and
/// therefore its rounding — cannot move. Every other width takes the generic extractor.
private let unpackHeader = """
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

/// The routed expert MLP of a one-token step in two dispatches over quantized weights,
/// where gather_qmm plus the elementwise chain needs five: the first computes
/// silu(gate) · up while the dots are still in registers, the second folds the routing
/// weights, the sum over experts and the residual into the down projection. The inner
/// loop is mlx's qmv_fast shape — two simdgroups of four rows, the activations cached
/// per lane and reused across rows.
///
/// `gateUpAct` wants the gate and up stacks row-interleaved ([g₀,u₀,g₁,u₁,…], done at
/// load), so each simdgroup's four rows are two (gate, up) pairs and the activation
/// needs no cross-simdgroup traffic. Every op boundary of the chain it replaces rounds
/// to T in place: gate and up after the dot, the sigmoid, both products, each
/// weighted product, the expert sum, the residual add.
private let gateUpActSource = """
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

private let downCombineSource = """
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
        size_t base = (size_t)IDX[e] * rows + row0;
        const device uint8_t* ws = (const device uint8_t*)W
            + base * kw_bytes + lane * bytes_per_thread;
        const device T* sl = S + base * kg + lane / lanes_per_group;
        const device T* bl = Bs + base * kg + lane / lanes_per_group;
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

private let gateUpActKernel = MLXFast.metalKernel(
    name: "moe_gateup_act", inputNames: ["X", "W", "S", "Bs", "IDX", "N", "KD", "GSIZE"],
    outputNames: ["Y"], source: gateUpActSource, header: unpackHeader)

private let downCombineKernel = MLXFast.metalKernel(
    name: "moe_down_combine",
    inputNames: ["ACT", "W", "S", "Bs", "IDX", "WTS", "RES", "N", "KD", "GSIZE"],
    outputNames: ["Y"], source: downCombineSource, header: unpackHeader)

/// Whether the fused pair can run this MoE. The reduction has to tile: sixteen values per
/// lane on the gate/up side, eight on the down side, and a lane's values have to sit
/// inside one quantization group. Nothing here is a width — 4-bit and 6-bit checkpoints
/// take the same path.
func moeGemvApplies(hidden: Int, inner: Int, gateUpGroup: Int, downGroup: Int) -> Bool {
    func tiles(_ kdim: Int, _ group: Int, _ perLane: Int) -> Bool {
        kdim % (perLane * 32) == 0 && group % perLane == 0 && kdim % group == 0
    }
    return tiles(hidden, gateUpGroup, 16) && tiles(inner, downGroup, 8)
        && inner % 4 == 0 && hidden % 8 == 0
}

/// One token through the routed, row-interleaved gate and up stacks, activation
/// applied: x [hidden], weight [experts, 2·inner, hidden·bits/32] uint32 with its scales
/// and biases, indices [k] — returns silu(gate) · up as [k, inner].
func moeGateUpAct(
    _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
    indices: MLXArray, groupSize: Int, bits: Int
) -> MLXArray {
    let inner = weight.dim(1) / 2
    let kdim = weight.dim(2) * 32 / bits
    precondition(bits <= 8 && kdim % 512 == 0 && groupSize % 16 == 0 && inner % 4 == 0)
    return gateUpActKernel(
        [x, weight, scales, biases, indices,
         MLXArray(Int32(inner)), MLXArray(Int32(kdim)), MLXArray(Int32(groupSize))],
        template: [("T", x.dtype), ("BITS", bits)],
        grid: (64, 2 * inner / 8, indices.dim(0)),
        threadGroup: (64, 1, 1),
        outputShapes: [[indices.dim(0), inner]],
        outputDTypes: [x.dtype])[0]
}

/// The routed down projection, routing weights, expert sum and residual in one dispatch:
/// act [k, inner], weight [experts, hidden, inner·bits/32] uint32 with scales and biases,
/// routing [k], residual [hidden] — returns the block's output [hidden].
func moeDownCombine(
    _ act: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
    indices: MLXArray, routing: MLXArray, residual: MLXArray, groupSize: Int, bits: Int
) -> MLXArray {
    let hidden = weight.dim(1)
    let kdim = weight.dim(2) * 32 / bits
    precondition(bits <= 8 && kdim % 256 == 0 && groupSize % 8 == 0 && hidden % 8 == 0)
    return downCombineKernel(
        [act, weight, scales, biases, indices, routing, residual,
         MLXArray(Int32(hidden)), MLXArray(Int32(kdim)), MLXArray(Int32(groupSize))],
        template: [("T", act.dtype), ("BITS", bits), ("TOPK", indices.dim(0))],
        grid: (64, hidden / 8, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[hidden]],
        outputDTypes: [act.dtype])[0]
}
