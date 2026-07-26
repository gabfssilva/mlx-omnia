"""The gated delta rule (DeltaNet recurrence) in one dispatch.

Ported verbatim from the Swift kernel (`.legacy/Sources/Sideros/Core/GatedDelta.swift`),
itself a port of mlx-lm's `gated_delta_step`: a simdgroup per (value head, value dim)
walks the T tokens sequentially, carrying its `[Dk]` slice of the state in float32
registers. Per token the state decays by `g`, the key reads the old memory, `beta`
scales the delta against the value, the key writes it back and the query projects the
state out. Key heads are broadcast over the value heads inside the kernel
(`hk = hv / (Hv/Hk)`, i.e. interleaved, the same mapping `mx.repeat` gives), so no
repeated q/k is ever materialized.

Layout note: the state here is `[B, Hv, Dv, Dk]` — Dk innermost, which is what makes
the per-lane slice contiguous. The ops path in `models/qwen3_5.py` carries it as
`[B, Hv, Dk, Dv]`; the two are a transpose of each other.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_SOURCE = """
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

_KERNEL = metal_kernel(
    name="gated_delta_step",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out"],
    source=_SOURCE,
)


def gated_delta_applies(
    key_dim: int, key_heads: int, value_heads: int, value_dim: int
) -> bool:
    """Key dim a multiple of 32 (one lane owns `Dk/32` contiguous entries), the value
    heads an exact multiple of the key heads (the in-kernel broadcast), and the value dim
    a multiple of the 4 rows the threadgroup below tiles the grid's y axis with."""
    return key_dim % 32 == 0 and value_heads % key_heads == 0 and value_dim % 4 == 0


def gated_delta(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
) -> tuple[mx.array, mx.array]:
    """One dispatch over the whole sequence.

    `q`, `k` are `[B, T, Hk, Dk]` already l2-normalized **and q already scaled**;
    `v` is `[B, T, Hv, Dv]`; `g` `[B, T, Hv]` the decay **past the exp** (not the log
    decay the ops path takes) and `beta` `[B, T, Hv]` the write strength; `state`
    `[B, Hv, Dv, Dk]` float32. Returns the mixed values in `q`'s dtype and the advanced
    state.
    """
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    assert gated_delta_applies(key_dim, key_heads, value_heads, value_dim)

    y, state_out = _KERNEL(
        inputs=[q, k, v, g, beta, state, mx.array(length, dtype=mx.int32)],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", key_heads),
            ("Hv", value_heads),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, length, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
    return y, state_out
