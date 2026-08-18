"""The gated delta rule (DeltaNet recurrence) in one dispatch.

Ported verbatim from the Swift kernel (`.legacy/Sources/Omnia/Core/GatedDelta.swift`),
itself a port of the reference implementation: a simdgroup per (value head, value dim)
walks the T tokens sequentially, carrying its `[Dk]` slice of the state in float32
registers. Per token the state decays by `g`, the key reads the old memory, `beta`
scales the delta against the value, the key writes it back and the query projects the
state out. Key heads are broadcast over the value heads inside the kernel
(`hk = hv / (Hv/Hk)`, i.e. interleaved, the same mapping `mx.repeat` gives), so no
repeated q/k is ever materialized.

Layout note: the state here is `[B, Hv, Dv, Dk]` — Dk innermost, which is what makes
the per-lane slice contiguous. The ops path in `default.py` carries it as
`[B, Hv, Dk, Dv]`; the two are a transpose of each other.

Two decays share the walk. A DeltaNet's `g` is one number per (token, value head) and
multiplies the whole state; KDA's is one per key channel, so the same lane that owns
`state[i]` owns its decay. That is the only difference between the two variants below —
where `g` is indexed and how far it advances per token — and it is compiled twice rather
than branched, since the index is a constant per dispatch.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.gated_delta.kernel import GatedDeltaStrategy
from mlx_omnia.engine.core.mxcompat import metal_kernel

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

    // beta: [B, T, Hv]
    __G_SETUP__
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * __G_ACCESS__;
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
      __G_ADVANCE__
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""


def _source(setup: str, access: str, advance: str) -> str:
    """`g`'s indexing, the only thing the two variants disagree on."""
    return (
        _SOURCE.replace("__G_SETUP__", setup)
        .replace("__G_ACCESS__", access)
        .replace("__G_ADVANCE__", advance)
    )


_PER_HEAD_SOURCE = _source("auto g_ = g + b_idx * T * Hv;", "g_[hv_idx]", "g_ += Hv;")
"""The per-head variant with its three holes filled — the text this kernel is compiled from,
and therefore the only text a source mutation can be written against. `_SOURCE` is the
template: two of the fragments a mutation names appear only once the holes are closed, and a
kernel built from the template itself does not compile at all."""

_KERNEL = metal_kernel(
    name="gated_delta_step",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out"],
    source=_PER_HEAD_SOURCE,
)

_KERNEL_PER_CHANNEL = metal_kernel(
    name="gated_delta_step_vec",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out"],
    source=_source(
        "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;", "g_[s_idx]", "g_ += Hv * Dk;"
    ),
)


_TRACE_SOURCE = """
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

    // states_seq: [B, T, Hv, Dv, Dk] — the state after every token, each lane its slice
    auto seq_ = states_seq + ((b_idx * T * Hv) + hv_idx) * Dv * Dk + dv_idx * Dk;

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
        seq_[s_idx] = static_cast<StT>(state[i]);
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
      seq_ += Hv * Dv * Dk;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""

_KERNEL_TRACE = metal_kernel(
    name="gated_delta_step_trace",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out", "states_seq"],
    source=_TRACE_SOURCE,
)


def gated_delta_trace(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """The same walk as `gated_delta`, also writing the state after every token.

    The extra output is what a compiled speculative verify rewinds by: keeping `kept`
    rows is picking `states_seq[:, kept - 1]` instead of replaying the layer. The stores
    are the registers cast to `StT` exactly as the final state is, so slot `T - 1` and
    `state_out` are the same bits — the walk itself is unchanged and `y` rounds
    identically to `gated_delta`'s. Per-head decay only: the per-channel (KDA) variant
    has no speculative caller.
    """
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    assert gated_delta_applies(key_dim, key_heads, value_heads, value_dim)
    assert g.ndim == 3, "per-channel decay has no trace variant"

    y, state_out, states_seq = _KERNEL_TRACE(
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
        output_shapes=[
            (batch, length, value_heads, value_dim),
            state.shape,
            (batch, length, value_heads, value_dim, key_dim),
        ],
        output_dtypes=[q.dtype, state.dtype, state.dtype],
    )
    return y, state_out, states_seq


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
    `v` is `[B, T, Hv, Dv]`; `g` the decay **past the exp** (not the log decay the ops
    path takes), either `[B, T, Hv]` per head or `[B, T, Hv, Dk]` per key channel; and
    `beta` `[B, T, Hv]` the write strength; `state` `[B, Hv, Dv, Dk]` float32. Returns
    the mixed values in `q`'s dtype and the advanced state.
    """
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    assert gated_delta_applies(key_dim, key_heads, value_heads, value_dim)

    kernel = _KERNEL_PER_CHANNEL if g.ndim == 4 else _KERNEL
    y, state_out = kernel(
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


@dataclass(frozen=True)
class FusedGatedDelta(GatedDeltaStrategy):
    """The whole recurrence in one dispatch, for the shapes the kernel tiles."""

    @classmethod
    def build(
        cls,
        *,
        key_dim: int,
        key_heads: int,
        value_heads: int,
        value_dim: int,
        enabled: bool,
    ) -> Self | None:
        if not enabled or not gated_delta_applies(
            key_dim, key_heads, value_heads, value_dim
        ):
            return None
        return cls()

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return gated_delta(q, k, v, g, beta, state)
