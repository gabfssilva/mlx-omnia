"""The SSD selective scan's one-token step, in one dispatch.

Ported from the reference implementation: a simdgroup per
`(batch, head)` tiles the state dimension, carrying its `[head_dim, state_size]`
slice in float32 registers. Per token: `A = -exp(A_log)`, `dA = exp(A·dt)`,
`state = dA·state + dt·B·x`, `y = state·C + D·x`. The group broadcast from
`n_groups` to `num_heads` happens in-kernel via the group offset
(`g_idx = n / G`), so no `repeat` is materialized. `compute_dt` (softplus + clamp)
runs on the host before the dispatch, the same split the reference implementation takes.

Layout: the state is `[B, H, Dh, Ds]` — `Ds` innermost, which is what makes the
per-lane slice contiguous. The ops path in `ssm/default.py` carries the same
layout; unlike `gated_delta` there is no transpose between the two.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.mxcompat import metal_kernel

_SOURCE = """
    auto n = thread_position_in_grid.z;
    auto h_idx = n % H;
    auto g_idx = n / G;
    constexpr int n_per_t = Ds / 32;

    auto x = X + n * Dh;
    out += n * Dh;
    auto i_state = state_in + n * Dh * Ds;
    auto o_state = state_out + n * Dh * Ds;

    // C and B have shape [batch, group, state_dim] (flattened);
    // offset by group — the in-kernel broadcast, no repeat materialized.
    auto C_ = C + g_idx * Ds;
    auto B_ = B + g_idx * Ds;

    auto ds_idx = thread_position_in_threadgroup.x;
    auto d_idx = thread_position_in_grid.y;

    auto dt_ = static_cast<float>(dt[n]);
    auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
    auto dA = fast::exp(A * dt_);

    float acc = 0.0;
    auto x_ = static_cast<float>(x[d_idx]);

    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * ds_idx + i;
        auto idx = d_idx * Ds + s_idx;
        auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
        auto state = dA * i_state[idx] + dB_by_x;
        o_state[idx] = static_cast<U>(state);
        acc += state * C_[s_idx];
    }
    acc = simd_sum(acc);
    if (thread_index_in_simdgroup == 0) {
        out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
    }
"""

_KERNEL = metal_kernel(
    name="ssm_step",
    input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
    output_names=["out", "state_out"],
    source=_SOURCE,
)


def ssm_step_applies(state_size: int, num_heads: int, n_groups: int) -> bool:
    """State size a multiple of 32 (one lane owns `Ds/32` contiguous entries) and
    the heads an exact multiple of the groups (the in-kernel broadcast)."""
    return state_size % 32 == 0 and num_heads % n_groups == 0


def _compute_dt(
    dt: mx.array, dt_bias: mx.array, time_step_limit: tuple[float, float]
) -> mx.array:
    """softplus(dt + dt_bias) in float32, clamped to the time-step limit."""
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def ssm_step(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: tuple[float, float],
) -> tuple[mx.array, mx.array]:
    """One decode step of the SSD recurrence.

    `x` is `[B, 1, H, Dh]` (the post-conv, post-reshape hidden);
    `A_log`, `D`, `dt_bias` are `[H]` (float32);
    `B`, `C` are `[B, 1, G, Ds]` (group-broadcast happens in-kernel);
    `dt` is `[B, 1, H]` (pre-softplus — `compute_dt` runs inside);
    `state` is `[B, H, Dh, Ds]` float32. Returns the mixed output `[B, 1, H, Dh]`
    in `x`'s dtype and the advanced state.
    """
    batch = x.shape[0]
    num_heads = x.shape[2]
    head_dim = x.shape[3]
    n_groups = B.shape[2]
    state_size = B.shape[3]
    assert ssm_step_applies(state_size, num_heads, n_groups)

    dt_processed = _compute_dt(dt, dt_bias, time_step_limit)
    heads_per_group = num_heads // n_groups

    out, state_out = _KERNEL(
        inputs=[x, A_log, B, C, D, dt_processed, state],
        template=[
            ("T", x.dtype),
            ("U", state.dtype),
            ("Dh", head_dim),
            ("Ds", state_size),
            ("H", num_heads),
            ("G", heads_per_group),
        ],
        grid=(32, head_dim, batch * num_heads),
        threadgroup=(32, 8, 1),
        output_shapes=[(batch, 1, num_heads, head_dim), state.shape],
        output_dtypes=[x.dtype, state.dtype],
    )
    return out, state_out
