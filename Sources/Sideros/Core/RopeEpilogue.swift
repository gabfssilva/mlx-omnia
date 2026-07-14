import MLX

/// The per-head tail of a one-token qkv projection — rms-norm each q and k head, then
/// rotate — in one dispatch where the op chain needs four (two norms, two ropes). One
/// threadgroup per q or k head; v needs neither and stays a free slice of the fused
/// projection. The norm's output rounds to T before the rotation reads it, as the
/// tensor between the two ops would.
private let epilogueSource = """
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    bool isq = head < (uint)QHEADS;
    uint base = isq ? head * HDIM : (uint)QHEADS * HDIM + (head - (uint)QHEADS) * HDIM;

    threadgroup float part[HDIM / 32];
    float v = (float)F[base + tid];
    float g = simd_sum(v * v);
    if (lane == 0) part[sg] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumsq = 0.0f;
    for (uint j = 0; j < HDIM / 32; j++) sumsq += part[j];
    float hr = metal::rsqrt(sumsq / (float)HDIM + EPS[0]);

    threadgroup float normed[HDIM];
    const device T* nw = isq ? QNW : KNW;
    float vn = (float)(T)(v * hr * (float)nw[tid]);
    normed[tid] = vn;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = tid % (HDIM / 2);
    bool hi = tid >= (uint)HDIM / 2;
    float other = normed[hi ? tid - HDIM / 2 : tid + HDIM / 2];
    float angle = (float)OFF[0] * FREQ[i];
    float c = metal::cos(angle), sn = metal::sin(angle);
    float rot = hi ? (other * sn + vn * c) : (vn * c - other * sn);
    if (isq) Q[head * HDIM + tid] = (T)rot;
    else K[(head - (uint)QHEADS) * HDIM + tid] = (T)rot;
    """

private let epilogueKernel = MLXFast.metalKernel(
    name: "rope_epilogue", inputNames: ["F", "QNW", "KNW", "OFF", "FREQ", "EPS"],
    outputNames: ["Q", "K"], source: epilogueSource)

/// fused [queryHeads·headDim + 2·kvHeads·headDim] is the qkv projection of one token;
/// offset [1] int32 is the rotary position; freqs [headDim/2] float32 holds
/// base^(−2i/headDim). Returns q [queryHeads·headDim] and k [kvHeads·headDim], both
/// head-major — a reshape away from attention's layout.
func ropeEpilogue(
    _ fused: MLXArray, queryHeads: Int, kvHeads: Int, headDim: Int,
    qNorm: MLXArray, kNorm: MLXArray, offset: MLXArray, freqs: MLXArray, eps: MLXArray
) -> (queries: MLXArray, keys: MLXArray) {
    precondition(headDim % 32 == 0)
    let out = epilogueKernel(
        [fused, qNorm, kNorm, offset, freqs, eps],
        template: [("T", fused.dtype), ("QHEADS", queryHeads), ("HDIM", headDim)],
        grid: ((queryHeads + kvHeads) * headDim, 1, 1),
        threadGroup: (headDim, 1, 1),
        outputShapes: [[queryHeads * headDim], [kvHeads * headDim]],
        outputDTypes: [fused.dtype, fused.dtype])
    return (out[0], out[1])
}
