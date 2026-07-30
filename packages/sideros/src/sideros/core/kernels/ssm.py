"""Mamba2 SSD: the decode kernel (reused from ``ssm_step``) plus the chunked
prefill scan and the dispatch function.

The per-token Metal kernel lives in ``core/kernels/ssm_step.py`` (ported from
mlx-lm's ``ssm_kernel``). This module adds the chunked SSD surrogate-attention
prefill (``ssm_attn``, pure mlx ops) and the ``ssm_update`` dispatch: T=1 +
state + GPU → kernel; else ``ssm_attn``. Also re-exports ``ssm_step`` and
``ssm_applies`` so the model file imports from one place.

No ``.legacy/`` Swift version exists — the kernel comes from mlx-lm verbatim.
"""

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.ssm_step import ssm_step as _ssm_step
from sideros.core.kernels.ssm_step import ssm_step_applies as _ssm_step_applies


def _compute_dt(
    dt: mx.array, dt_bias: mx.array, time_step_limit: tuple[float, float]
) -> mx.array:
    """softplus(dt + dt_bias) in fp32, clamped to the time-step limit."""
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias.astype(mx.float32))
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def ssm_applies(head_dim: int, d_state: int, num_heads: int, num_groups: int) -> bool:
    """The kernel tiles ``d_state`` across 32 lanes and broadcasts groups to
    heads in-kernel: ``d_state`` must be a multiple of 32 and ``num_heads`` an
    exact multiple of ``num_groups``."""
    return _ssm_step_applies(d_state, num_heads, num_groups)


def ssm_step(
    hidden: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple[mx.array, mx.array]:
    """One-token SSD step via the Metal kernel. Delegates to ``ssm_step.ssm_step``.

    ``hidden`` is ``[B, 1, H, Dh]``; ``B``/``C`` are ``[B, 1, G, Ds]``;
    ``A_log``/``D``/``dt_bias`` are ``[H]``; ``dt`` is ``[B, 1, H]`` (pre-softplus
    — ``compute_dt`` runs inside); ``state`` is ``[B, H, Dh, Ds]`` fp32.
    """
    return _ssm_step(hidden, A_log, B, C, D, dt, dt_bias, state, time_step_limit)


def _segsum(x: mx.array) -> mx.array:
    """Stable segment sum: cumsum of the strictly-lower-triangular part, with the
    upper triangle (including diagonal) set to -inf so ``exp`` zeroes it."""
    length = x.shape[-1]
    expanded = mx.repeat(x[..., None], length, axis=-1)
    masked = mx.tril(expanded, -1)
    segsum = mx.cumsum(masked, axis=-2)
    keep = mx.tril(mx.ones((length, length), dtype=mx.bool_), 0)
    return mx.where(keep, segsum, -float("inf"))


def ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    *,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
    step: int = 256,
) -> tuple[mx.array, mx.array]:
    """Chunked SSD prefill (pure mlx ops). Ported from mlx-lm's ``ssm_attn``.

    ``x`` ``[B, L, H, Dh]``; ``B``/``C`` ``[B, L, G, Ds]``; ``A_log`` ``[H]``;
    ``D`` ``[H]``; ``dt`` ``[B, L, H]`` (pre softplus+bias); ``state``
    ``[B, H, Dh, Ds]`` fp32 or None.
    """
    batch, length, heads, head_dim = x.shape
    _, _, groups, d_state = B.shape
    repeats = heads // groups

    dt_processed = _compute_dt(dt, dt_bias, time_step_limit)
    A = -mx.exp(A_log).astype(dt_processed.dtype)
    dtA = dt_processed * A.reshape(1, 1, -1)
    dtx = dt_processed.reshape(batch, length, heads, 1) * x.astype(dt_processed.dtype)

    def _chunk(
        dtx: mx.array,
        dtA: mx.array,
        B: mx.array,
        C: mx.array,
        state: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        s = dtx.shape[1]
        B_t = mx.transpose(B, (0, 2, 3, 1))
        cb = mx.swapaxes(C, 1, 2) @ B_t
        cb = mx.repeat(cb, repeats, axis=1)
        decay = mx.exp(_segsum(dtA.swapaxes(1, 2)))
        surrogate = mx.tril(cb * decay, 0)
        y = surrogate @ dtx.swapaxes(1, 2)
        y = mx.swapaxes(y, 1, 2)

        decay_last = decay[:, :, -1:, :].transpose(0, 3, 1, 2)
        B_rep = mx.repeat(B_t, repeats, axis=1).swapaxes(2, 3)
        dtx_decay = (dtx * decay_last).swapaxes(1, 2).swapaxes(2, 3)
        next_state = dtx_decay @ B_rep

        if state is not None:
            exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
            next_state = next_state + exp_dtA_cumsum[:, -1, :, None, None] * state
            C_r = C.reshape(batch, s, groups, 1, d_state, 1)
            y_prev = (
                state.reshape((batch, 1, groups, repeats, head_dim, d_state))
                @ C_r
            ).squeeze(-1).flatten(2, 3)
            y = y + exp_dtA_cumsum[..., None] * y_prev

        return y.astype(x.dtype), next_state

    ys: list[mx.array] = []
    for i in range(0, length, step):
        y, state = _chunk(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
        )
        ys.append(y)
    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, heads, 1)
    assert state is not None
    return y, state


def ssm_update(
    hidden: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    *,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
    step: int = 256,
    d_state: int = 0,
    groups: int = 0,
) -> tuple[mx.array, mx.array]:
    """Dispatch: T=1 + state + GPU → Metal kernel; else chunked prefill.

    ``hidden`` is ``[B, L, intermediate]`` (will be reshaped to
    ``[B, L, H, Dh]``); ``B``/``C`` are ``[B, L, G*Ds]`` (will be reshaped to
    ``[B, L, G, Ds]``); ``dt`` ``[B, L, H]`` (pre softplus+bias). ``d_state``
    and ``groups`` are the config's, passed by the caller since the shape alone
    is ambiguous (``G*Ds`` decomposes in one way given the config).
    """
    batch, seq_len, intermediate = hidden.shape
    heads = A_log.shape[0]
    head_dim = intermediate // heads
    if d_state == 0 or groups == 0:
        raise ValueError("ssm_update requires d_state and groups from the config")
    if B.ndim == 3:
        b = B.reshape(batch, seq_len, groups, d_state)
        c = C.reshape(batch, seq_len, groups, d_state)
    else:
        b, c = B, C

    hidden_r = hidden.reshape(batch, seq_len, heads, head_dim)

    if (
        seq_len == 1
        and state is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and ssm_applies(head_dim, d_state, heads, groups)
    ):
        return ssm_step(
            hidden_r, A_log, b, c, D, dt, dt_bias, state, time_step_limit
        )

    return ssm_attn(
        hidden_r, A_log, b, c, D, dt, dt_bias, state,
        time_step_limit=time_step_limit, step=step,
    )
