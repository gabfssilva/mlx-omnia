import MLX

/// MLX's Metal matmul is bandwidth-optimal at one row (gemv) and at large tiles, but a
/// 2–8 row bf16 gemm — exactly what speculative verification feeds it — runs the big
/// tile kernel at ~60% of the gemv's bandwidth, flat across that whole range. This
/// kernel closes the gap: one simdgroup streams two rows of W with vectorized loads,
/// keeping M accumulators per lane, so the weights are read once at gemv-like
/// coalescing whatever the row count.
///
/// y[M, N] = x[M, K] @ W[N, K]ᵀ — W in checkpoint layout, rows contiguous.
private let source = """
    uint pair = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint k4 = (uint)K / 4;
    if (n0 >= n) return;

    const device bfloat4* w0 = (const device bfloat4*)(W + (size_t)n0 * K);
    const device bfloat4* w1 = (const device bfloat4*)(W + (size_t)(n0 + 1) * K);
    const device bfloat4* xv = (const device bfloat4*)X;

    float4 acc0[M];
    float4 acc1[M];
    for (int m = 0; m < M; m++) { acc0[m] = float4(0.0f); acc1[m] = float4(0.0f); }
    for (uint i = lane; i < k4; i += 32) {
        float4 wv0 = float4(w0[i]);
        float4 wv1 = float4(w1[i]);
        for (int m = 0; m < M; m++) {
            float4 xm = float4(xv[m * k4 + i]);
            acc0[m] += xm * wv0;
            acc1[m] += xm * wv1;
        }
    }
    for (int m = 0; m < M; m++) {
        float t0 = simd_sum(acc0[m].x + acc0[m].y + acc0[m].z + acc0[m].w);
        float t1 = simd_sum(acc1[m].x + acc1[m].y + acc1[m].z + acc1[m].w);
        if (lane == 0) {
            Y[(size_t)m * n + n0] = (bfloat16_t)t0;
            Y[(size_t)m * n + n0 + 1] = (bfloat16_t)t1;
        }
    }
    """

private let kernel = MLXFast.metalKernel(
    name: "skinny_gemm", inputNames: ["X", "W", "N", "K"], outputNames: ["Y"],
    source: source)

/// The rows the kernel beats plain matmul on. One row is the gemv fast path already;
/// past eight, accumulator pressure loses to the tile kernel.
let skinnyRows = 2...8

/// `weight` transposed to checkpoint layout [N, K] — a view, contiguous only when the
/// linear was loaded from the standard HF layout, which every bf16 model here is.
func skinnyMatmul(_ x: MLXArray, weight: MLXArray) -> MLXArray {
    let (rows, k) = (x.dim(-2), x.dim(-1))
    let n = weight.dim(0)
    let y = kernel(
        [x, weight, MLXArray(Int32(n)), MLXArray(Int32(k))],
        template: [("M", rows)],
        grid: (32, (n + 1) / 2, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[rows, n]],
        outputDTypes: [x.dtype])[0]
    return y.reshaped(x.shape.dropLast() + [n])
}
