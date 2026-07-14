import MLX

/// The gated delta rule in one dispatch, ported from mlx-lm's `gated_delta_step`: a
/// simdgroup per (value head, 4 value dims) walks the T tokens sequentially, carrying
/// its slice of the [Dv, Dk] state in registers at float32. Per token: the state decays
/// by g, the key reads the old memory, beta scales the delta against the value, the key
/// writes it back, and the query projects the state out. Key heads are broadcast over
/// the value heads (hk = hv / (Hv/Hk)), so no repeat is ever materialized.
private let gatedDeltaSource = """
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    // g, beta: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
    """

private let gatedDeltaKernel = MLXFast.metalKernel(
    name: "gated_delta_step", inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
    outputNames: ["y", "state_out"], source: gatedDeltaSource)

/// q, k [B, T, Hk, Dk] and v [B, T, Hv, Dv] already normalized and scaled; g [B, T, Hv]
/// the decay past the exp, beta [B, T, Hv] the write strength, state [B, Hv, Dv, Dk]
/// float32. Returns the mixed values in q's dtype and the advanced state.
func gatedDelta(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
) -> (y: MLXArray, state: MLXArray) {
    let (batch, length, keyHeads, keyDim) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let (valueHeads, valueDim) = (v.dim(2), v.dim(3))
    precondition(keyDim % 32 == 0 && valueHeads % keyHeads == 0)

    let out = gatedDeltaKernel(
        [q, k, v, g, beta, state, MLXArray(Int32(length))],
        template: [
            ("InT", q.dtype), ("StT", state.dtype),
            ("Dk", keyDim), ("Dv", valueDim), ("Hk", keyHeads), ("Hv", valueHeads),
        ],
        grid: (32, valueDim, batch * valueHeads),
        threadGroup: (32, 4, 1),
        outputShapes: [[batch, length, valueHeads, valueDim], state.shape],
        outputDTypes: [q.dtype, state.dtype])
    return (out[0], out[1])
}
