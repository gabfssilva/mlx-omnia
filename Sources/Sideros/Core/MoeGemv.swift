import MLX

/// MLX's gather matmul streams a one-token routed expert MLP at ~450 GB/s on the
/// M5 Max, while a hand-rolled row-pair gemv with the expert indirection sustains
/// ~700: the tile kernel it dispatches at m=1 wastes half the bandwidth. These two
/// kernels run the whole routed MLP in two dispatches — the second computes
/// silu(gate) · up on the fly while streaming the down weights, and folds the
/// routing weight into the output, so nothing elementwise runs between them.
///
/// Both take the expert stacks in checkpoint layout [experts, out, in], rows
/// contiguous, and a flat index vector naming the routed experts.

/// Y[e, n] = dot(X, W[IDX[e], n, :]) — one simdgroup per two rows of one expert.
private let gateUpSource = """
    uint pair = thread_position_in_grid.y;
    uint expert = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint k4 = (uint)K / 4;
    if (n0 >= n) return;

    size_t base = (size_t)IDX[expert] * n * (uint)K;
    const device vec<T, 4>* w0 = (const device vec<T, 4>*)(W + base + (size_t)n0 * K);
    const device vec<T, 4>* w1 = (const device vec<T, 4>*)(W + base + (size_t)(n0 + 1) * K);
    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xm = float4(xv[i]);
        acc0 += xm * float4(w0[i]);
        acc1 += xm * float4(w1[i]);
    }
    float t0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float t1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0) {
        Y[(size_t)expert * n + n0] = (T)t0;
        Y[(size_t)expert * n + n0 + 1] = (T)t1;
    }
    """

/// As above, but X is the gate-up output: row e holds [gate | up] and the kernel
/// streams silu(gate) · up as it loads, then scales the dot by WS[e].
private let downSource = """
    uint pair = thread_position_in_grid.y;
    uint expert = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint k4 = (uint)K / 4;
    if (n0 >= n) return;

    size_t base = (size_t)IDX[expert] * n * (uint)K;
    const device vec<T, 4>* w0 = (const device vec<T, 4>*)(W + base + (size_t)n0 * K);
    const device vec<T, 4>* w1 = (const device vec<T, 4>*)(W + base + (size_t)(n0 + 1) * K);
    const device vec<T, 4>* gv = (const device vec<T, 4>*)(GU + (size_t)expert * 2 * (uint)K);
    const device vec<T, 4>* uv = (const device vec<T, 4>*)(GU + ((size_t)expert * 2 + 1) * (uint)K);

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 g = float4(gv[i]);
        float4 xm = g / (1.0f + metal::exp(-g)) * float4(uv[i]);
        acc0 += xm * float4(w0[i]);
        acc1 += xm * float4(w1[i]);
    }
    float t0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float t1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0) {
        float scale = (float)WS[expert];
        Y[(size_t)expert * n + n0] = (T)(t0 * scale);
        Y[(size_t)expert * n + n0 + 1] = (T)(t1 * scale);
    }
    """

/// The whole routing decision in one dispatch: one threadgroup of E simdgroups.
/// Each simdgroup streams its expert's router row coalesced and simd-sums the dot
/// — one lane per row would expose full memory latency with nothing to hide it,
/// ~100 µs against this shape's ~7. The first simdgroup then walks simd_max k
/// times over the sigmoid scores and normalizes the winners' weights in place.
/// The unfused chain (gemv, sigmoid, argpartition, takeAlong, normalize) is ~10
/// tiny dependent kernels whose launch latency alone costs ~80 µs per layer.
private let routeSource = """
    uint sg = thread_position_in_threadgroup.x / 32;
    uint lane = thread_position_in_threadgroup.x % 32;
    uint k4 = (uint)K / 4;

    threadgroup float scores[E];
    const device vec<T, 4>* w = (const device vec<T, 4>*)(RW + (size_t)sg * K);
    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        acc += float4(xv[i]) * float4(w[i]);
    }
    float dot = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) {
        scores[sg] = 1.0f / (1.0f + metal::exp(-dot));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;

    float score = scores[lane];
    float selector = score + BIAS[lane];
    float picked[TOPK];
    float total = 0.0f;
    for (uint j = 0; j < TOPK; j++) {
        float best = simd_max(selector);
        uint winner = simd_min(selector == best ? lane : (uint)E);
        float s = simd_shuffle(score, (ushort)winner);
        picked[j] = s;
        total += s;
        if (lane == winner) selector = -INFINITY;
        if (lane == 0) OI[j] = winner;
    }
    if (lane < TOPK) {
        float w = picked[lane];
        OW[lane] = (T)((NORM ? w / (total + 1e-6f) : w) * SCALE[0]);
    }
    """

private let routeKernel = MLXFast.metalKernel(
    name: "moe_route", inputNames: ["X", "RW", "BIAS", "SCALE", "K"],
    outputNames: ["OI", "OW"], source: routeSource)

private let gateUpKernel = MLXFast.metalKernel(
    name: "moe_gate_up", inputNames: ["X", "W", "IDX", "N", "K"], outputNames: ["Y"],
    source: gateUpSource)

private let downKernel = MLXFast.metalKernel(
    name: "moe_down", inputNames: ["GU", "W", "IDX", "WS", "N", "K"], outputNames: ["Y"],
    source: downSource)

/// One token's routing: x [hidden], router [experts, hidden], bias [experts]
/// float32, scale [] float32 — returns expert indices [k] and their routing
/// weights [k], normalized when `normalized`. One simdgroup carries every expert.
func moeRoute(
    _ x: MLXArray, router: MLXArray, bias: MLXArray, scale: MLXArray, k: Int,
    normalized: Bool
) -> (indices: MLXArray, weights: MLXArray) {
    let experts = router.dim(0)
    precondition(experts <= 32 && router.dim(1) % 4 == 0)
    let out = routeKernel(
        [x, router, bias, scale, MLXArray(Int32(router.dim(1)))],
        template: [("T", x.dtype), ("E", experts), ("TOPK", k), ("NORM", normalized)],
        grid: (experts * 32, 1, 1),
        threadGroup: (experts * 32, 1, 1),
        outputShapes: [[k], [k]],
        outputDTypes: [.uint32, x.dtype])
    return (out[0], out[1])
}

/// One token through the routed gate and up projections: x [hidden], weights
/// [experts, 2·inner, hidden], indices [k] — returns [k, 2·inner].
func moeGateUp(_ x: MLXArray, weights: MLXArray, indices: MLXArray) -> MLXArray {
    let (n, k) = (weights.dim(1), weights.dim(2))
    precondition(k % 4 == 0 && n % 2 == 0)
    return gateUpKernel(
        [x, weights, indices, MLXArray(Int32(n)), MLXArray(Int32(k))],
        template: [("T", x.dtype)],
        grid: (32, n / 2, indices.dim(0)),
        threadGroup: (32, 1, 1),
        outputShapes: [[indices.dim(0), n]],
        outputDTypes: [x.dtype])[0]
}

/// The routed down projection over the gate-up rows, each output row scaled by
/// its routing weight: gateUp [k, 2·inner], weights [experts, hidden, inner],
/// routing [k] — returns [k, hidden], ready to sum over k.
func moeDown(_ gateUp: MLXArray, weights: MLXArray, indices: MLXArray, routing: MLXArray)
    -> MLXArray
{
    let (n, k) = (weights.dim(1), weights.dim(2))
    precondition(k % 4 == 0 && n % 2 == 0)
    return downKernel(
        [gateUp, weights, indices, routing, MLXArray(Int32(n)), MLXArray(Int32(k))],
        template: [("T", gateUp.dtype)],
        grid: (32, n / 2, indices.dim(0)),
        threadGroup: (32, 1, 1),
        outputShapes: [[indices.dim(0), n]],
        outputDTypes: [gateUp.dtype])[0]
}
