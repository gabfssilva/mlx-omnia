import MLX

/// The gated short conv's one-token step in a single dispatch: each simdgroup
/// computes the B, C and x projections for two channels, applies the three causal
/// taps in float32 against the two window rows, gates, and emits the new window.
/// Unfused this is a gemv plus ~6 elementwise kernels; fused it streams the same
/// 25MB at the machine's small-gemv rate (~2.4x the compiled chain).
///
/// The in_proj weight comes in checkpoint layout [3·hidden, hidden] with the B,
/// C, x blocks stacked in that order; TAPS is the conv weight flattened [hidden·3];
/// WIN holds the previous two B·x rows.
private let convMixSource = """
    uint pair = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint c0 = pair * 2;
    uint h = (uint)HD;
    uint k4 = h / 4;
    if (c0 >= h) return;

    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;
    const device vec<T, 4>* wB0 = (const device vec<T, 4>*)(W + (size_t)c0 * h);
    const device vec<T, 4>* wB1 = (const device vec<T, 4>*)(W + (size_t)(c0 + 1) * h);
    const device vec<T, 4>* wC0 = (const device vec<T, 4>*)(W + (size_t)(h + c0) * h);
    const device vec<T, 4>* wC1 = (const device vec<T, 4>*)(W + (size_t)(h + c0 + 1) * h);
    const device vec<T, 4>* wX0 = (const device vec<T, 4>*)(W + (size_t)(2 * h + c0) * h);
    const device vec<T, 4>* wX1 = (const device vec<T, 4>*)(W + (size_t)(2 * h + c0 + 1) * h);

    float4 aB0 = float4(0.0f), aC0 = float4(0.0f), aX0 = float4(0.0f);
    float4 aB1 = float4(0.0f), aC1 = float4(0.0f), aX1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xm = float4(xv[i]);
        aB0 += xm * float4(wB0[i]); aB1 += xm * float4(wB1[i]);
        aC0 += xm * float4(wC0[i]); aC1 += xm * float4(wC1[i]);
        aX0 += xm * float4(wX0[i]); aX1 += xm * float4(wX1[i]);
    }
    float B[2], C[2], Xp[2];
    B[0] = simd_sum(aB0.x + aB0.y + aB0.z + aB0.w);
    B[1] = simd_sum(aB1.x + aB1.y + aB1.z + aB1.w);
    C[0] = simd_sum(aC0.x + aC0.y + aC0.z + aC0.w);
    C[1] = simd_sum(aC1.x + aC1.y + aC1.z + aC1.w);
    Xp[0] = simd_sum(aX0.x + aX0.y + aX0.z + aX0.w);
    Xp[1] = simd_sum(aX1.x + aX1.y + aX1.z + aX1.w);
    if (lane == 0) {
        for (uint r = 0; r < 2; r++) {
            uint c = c0 + r;
            // B·x rounded like the unfused elementwise: both factors and the
            // product pass through T before the float32 tap accumulation.
            float bx = (float)(T)((float)(T)B[r] * (float)(T)Xp[r]);
            float conv = (float)TAPS[c * 3 + 0] * (float)WIN[c]
                       + (float)TAPS[c * 3 + 1] * (float)WIN[h + c]
                       + (float)TAPS[c * 3 + 2] * bx;
            GATED[c] = (T)((float)(T)C[r] * (float)(T)conv);
            WOUT[c] = WIN[h + c];
            WOUT[h + c] = (T)bx;
        }
    }
    """

private let convMixKernel = MLXFast.metalKernel(
    name: "conv_mix", inputNames: ["X", "W", "TAPS", "WIN", "HD"],
    outputNames: ["GATED", "WOUT"], source: convMixSource)

/// x [hidden], weights [3·hidden, hidden], taps [hidden·3], window [2, hidden] —
/// returns the gated conv output [hidden] and the slid window [2, hidden].
func convMix(_ x: MLXArray, weights: MLXArray, taps: MLXArray, window: MLXArray)
    -> (gated: MLXArray, window: MLXArray)
{
    let hidden = x.dim(0)
    precondition(hidden % 4 == 0 && window.dim(0) == 2)
    let out = convMixKernel(
        [x, weights, taps, window, MLXArray(Int32(hidden))],
        template: [("T", x.dtype)],
        grid: (32, hidden / 2, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[hidden], [2, hidden]],
        outputDTypes: [x.dtype, x.dtype])
    return (out[0], out[1])
}
