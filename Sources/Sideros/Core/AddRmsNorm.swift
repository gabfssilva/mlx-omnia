import MLX

/// A residual add and the rms-norm that reads it, in one dispatch: the norm needs the
/// whole vector, so one threadgroup carries it — four values per thread, the sum of
/// squares reduced through threadgroup memory. Returns both the sum (the stream the
/// block's output adds to) and its normed view (what the next projection reads). The
/// add rounds to T before the squares, as the tensor between the ops would.
private let addNormSource = """
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    threadgroup float vals[HDIM];
    threadgroup float part[HDIM / 128];
    float local = 0.0f;
    for (uint j = 0; j < 4; j++) {
        uint i = tid * 4 + j;
        float a = (float)(T)((float)X[i] + (float)Op[i]);
        vals[i] = a;
        local += a * a;
    }
    float s = simd_sum(local);
    if (tid % 32 == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumsq = 0.0f;
    for (uint j = 0; j < HDIM / 128; j++) sumsq += part[j];
    float r = metal::rsqrt(sumsq / (float)HDIM + EPS[0]);
    for (uint j = 0; j < 4; j++) {
        uint i = tid * 4 + j;
        A[i] = (T)vals[i];
        HN[i] = (T)(vals[i] * r * (float)NW[i]);
    }
    """

private let addNormKernel = MLXFast.metalKernel(
    name: "add_rmsnorm", inputNames: ["X", "Op", "NW", "EPS"], outputNames: ["A", "HN"],
    source: addNormSource)

/// x and projected [hidden] are the residual stream and the branch joining it; weight
/// is the norm's; eps [1] float32. Returns (x + projected, rmsNorm of it).
func addRmsNorm(_ x: MLXArray, _ projected: MLXArray, weight: MLXArray, eps: MLXArray)
    -> (added: MLXArray, normed: MLXArray)
{
    let hidden = weight.size
    precondition(hidden % 128 == 0)
    let out = addNormKernel(
        [x, projected, weight, eps],
        template: [("T", x.dtype), ("HDIM", hidden)],
        grid: (hidden / 4, 1, 1),
        threadGroup: (hidden / 4, 1, 1),
        outputShapes: [[hidden], [hidden]],
        outputDTypes: [x.dtype, x.dtype])
    return (out[0], out[1])
}
