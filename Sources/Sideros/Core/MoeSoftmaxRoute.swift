import MLX

/// The pick of a one-token softmax routing in one dispatch where the op chain spends
/// five: `softmaxTopK` replaces softmax → argPartition → takeAlong → renormalize. The
/// logits still come from the stock quantized matmul — recomputing them in a fused
/// kernel rounds differently and flips which expert wins a near-tie, which moves a
/// whole expert's output, not a ulp.
///
/// One simdgroup, `EXPERTS / 32` scores per lane: 4 at 128 experts, 8 at 256. The kernel
/// reproduces the reference graph's roundings exactly: probabilities round to the logits'
/// dtype, and the kept `k` sum sequentially in that dtype ascending (mlx's argpartition
/// here is a stable ascending sort, so that is the order the reference sums in — and why
/// a tie at the cut goes to the higher index).
private let topKSource = """
    uint lane = thread_position_in_threadgroup.x;
    constexpr uint per_lane = EXPERTS / 32;

    float p[per_lane];
    float m = -INFINITY;
    for (uint i = 0; i < per_lane; i++) {
        p[i] = (float)L[lane + i * 32];
        m = metal::max(m, p[i]);
    }
    m = simd_max(m);
    float e = 0.0f;
    for (uint i = 0; i < per_lane; i++) {
        p[i] = metal::exp(p[i] - m);
        e += p[i];
    }
    float denom = simd_sum(e);
    for (uint i = 0; i < per_lane; i++) {
        p[i] = (float)(T)(p[i] / denom);
    }

    float pick[TOPK];
    for (uint j = 0; j < TOPK; j++) {
        float local = -1.0f;
        for (uint i = 0; i < per_lane; i++) {
            local = metal::max(local, p[i]);
        }
        float best = simd_max(local);
        // Descending, so a tie inside a lane goes to the higher slot; simd_max over the
        // candidates then takes the higher lane. Together: ties go to the higher index.
        int slot = -1;
        for (int i = (int)per_lane - 1; i >= 0; i--) {
            if (slot < 0 && p[i] == best) slot = i;
        }
        int cand = slot >= 0 ? (int)(lane + (uint)slot * 32) : -1;
        int winner = simd_max(cand);
        pick[j] = best;
        // Cleared by compare, not by p[slot]: a dynamic index would spill the scores out
        // of registers.
        for (uint i = 0; i < per_lane; i++) {
            if (winner == cand && (int)i == slot) p[i] = -1.0f;
        }
        if (lane == 0) OI[j] = (uint)winner;
    }
    float total = 0.0f;
    for (int j = TOPK - 1; j >= 0; j--) {
        total = (float)(T)(total + pick[j]);
    }
    if (lane < TOPK) OW[lane] = (T)(pick[lane] / total);
    """

private let topKKernel = MLXFast.metalKernel(
    name: "moe_softmax_topk", inputNames: ["L"], outputNames: ["OI", "OW"],
    source: topKSource)

/// One token's pick over the router's softmax scores: logits [experts] — returns the top
/// `k` expert indices, ties to the higher index, and their renormalized weights.
func softmaxTopK(_ logits: MLXArray, k: Int) -> (indices: MLXArray, weights: MLXArray) {
    precondition(logits.size % 32 == 0 && k <= 32)
    let out = topKKernel(
        [logits],
        template: [("T", logits.dtype), ("TOPK", k), ("EXPERTS", logits.size)],
        grid: (32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[k], [k]],
        outputDTypes: [.uint32, logits.dtype])
    return (out[0], out[1])
}
