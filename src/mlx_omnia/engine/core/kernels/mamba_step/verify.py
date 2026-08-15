"""The Mamba2 middle over a speculative verification's few rows, with per-token states.

The verification forward hands a recurrent layer T=2..4 rows, and a rejection needs the
state as it stood after the last accepted one. The replay answer runs the layer again;
this kernel makes the checkpoint free instead: the recurrence writes the state after
*every* token to its own slot, and the round's rewind is picking a slot — vllm's
spec-decode `selective_state_update` shape, in the head-per-threadgroup layout of
`fused.py`.

Same arithmetic as the T=1 path, token by token: the conv chain is `conv_step`'s, dt is
mlx's `logaddexp(x, 0)` in fp32, the recurrence is `ssm_step`'s per-row order. The
window slots are not produced here — a window is raw conv inputs, so the caller slices
them off the projection rows. The grouped gated norm stays outside, on the op chain,
over all T rows at once.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.mamba_step.fused import FusedMambaStep
from mlx_omnia.engine.core.kernels.mamba_step.norm import (
    gated_group_norm,
    gated_group_norm_applies,
)
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint n = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    constexpr uint inner = INNER;
    constexpr uint conv = CONVD;
    constexpr uint pw = INNER + CONVD + HEADS;
    constexpr uint dh = DH;
    constexpr uint ds = DS;
    constexpr uint hpg = HPG;
    constexpr uint k = K;
    constexpr uint tt = TT;
    constexpr uint npt = ds / 32;
    constexpr uint rows_per_sg = dh / 8;

    uint g = n / hpg;

    threadgroup float xc[tt][dh];
    threadgroup float bc[tt][ds];
    threadgroup float cc[tt][ds];
    threadgroup float dtv[tt];

    {
    #pragma clang fp contract(off)
    for (uint j = tid; j < tt * (dh + 2 * ds + 1); j += 256) {
        uint t = j / (dh + 2 * ds + 1);
        uint w = j % (dh + 2 * ds + 1);
        uint c = 0xffffffffu;
        threadgroup float* slot = nullptr;
        if (w < dh) {
            c = n * dh + w;
            slot = &xc[t][w];
        } else if (w < dh + ds) {
            c = inner + g * ds + (w - dh);
            slot = &bc[t][w - dh];
        } else if (w < dh + 2 * ds) {
            c = inner + GROUPS * ds + g * ds + (w - dh - ds);
            slot = &cc[t][w - dh - ds];
        } else {
            float v = (float)P[t * pw + inner + conv + n] + DTB[n];
            float mx_ = metal::max(v, 0.0f);
            float mn_ = metal::min(v, 0.0f);
            float sp = mx_ + ::log1p(metal::exp(mn_ - mx_));
            dtv[t] = metal::min(metal::max(sp, LIMS[0]), LIMS[1]);
            continue;
        }
        // The window at token t is the previous k-1 conv inputs, drawn from the cached
        // window for negative offsets and from the projection rows past them.
        float m = 0.0f;
        for (uint tap = 0; tap < k; tap++) {
            int src = (int)t - (int)(k - 1) + (int)tap;
            float value = src < 0
                ? (float)WIN[(uint)((int)(k - 1) + src) * conv + c]
                : (float)P[(uint)src * pw + inner + c];
            float p = (float)(T)(value * (float)TAPS[c * k + tap]);
            m = tap == 0 ? p : (float)(T)(m + p);
        }
        if (HAS_CBIAS) {
            m = (float)(T)(m + (float)CB[c]);
        }
        T mt = (T)m;
        T y = (T)1 / ((T)1 + metal::precise::exp(metal::abs(mt)));
        T s = (mt < (T)0) ? y : (T)1 - y;
        *slot = (float)(T)((T)m * s);
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float A = -metal::fast::exp(AL[n]);
    for (uint r = sg * rows_per_sg; r < (sg + 1) * rows_per_sg; r++) {
        size_t base = ((size_t)n * dh + r) * ds;
        float st[npt];
        for (uint i = 0; i < npt; ++i) {
            st[i] = SIN[base + npt * lane + i];
        }
        for (uint t = 0; t < tt; t++) {
            float dt_ = dtv[t];
            float dA = metal::fast::exp(A * dt_);
            float x_ = xc[t][r];
            float acc = 0.0f;
            size_t slot = (size_t)t * (size_t)HEADS * dh * ds + base;
            for (uint i = 0; i < npt; ++i) {
                uint s_idx = npt * lane + i;
                float dB_by_x = x_ * dt_ * bc[t][s_idx];
                st[i] = dA * st[i] + dB_by_x;
                SOUTS[slot + s_idx] = st[i];
                acc += st[i] * cc[t][s_idx];
            }
            acc = simd_sum(acc);
            if (lane == 0) {
                Y[t * inner + n * dh + r] = (T)(acc + x_ * DD[n]);
            }
        }
    }
"""

_KERNEL = metal_kernel(
    name="mamba_verify",
    input_names=["P", "WIN", "TAPS", "CB", "SIN", "AL", "DD", "DTB", "LIMS"],
    output_names=["Y", "SOUTS"],
    source=_SOURCE,
)


@dataclass(frozen=True)
class VerifyMambaStep:
    """Bound off a built `FusedMambaStep`: same leaves, same predicates, plus T."""

    taps: mx.array
    conv_bias: mx.array
    A_log: mx.array
    D: mx.array
    dt_bias: mx.array
    norm_weight: mx.array
    limits: mx.array
    eps_array: mx.array
    eps: float
    inner: int
    conv_dim: int
    kernel: int
    heads: int
    head_dim: int
    groups: int
    state_size: int
    has_bias: bool

    @classmethod
    def of(cls, fused: FusedMambaStep) -> Self:
        return cls(
            fused.taps, fused.conv_bias, fused.A_log, fused.D, fused.dt_bias,
            fused.norm_weight, fused.limits, fused.eps_array, fused.eps, fused.inner,
            fused.conv_dim,
            fused.kernel, fused.heads, fused.head_dim, fused.groups, fused.state_size,
            fused.has_bias,
        )

    def __call__(
        self, proj: mx.array, window: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array]:
        """proj [T, inner+conv_dim+heads] -> (normed [T, inner], states [T, H, Dh, Ds]).

        `T` must not exceed `kernel - 1` plus one; past that the in-kernel window
        indexing still holds, so the true bound is only threadgroup memory — 4 covers
        every verification block in use.
        """
        tokens = proj.shape[0]
        out, slots = _KERNEL(
            inputs=[
                proj, window, self.taps, self.conv_bias, state,
                self.A_log, self.D, self.dt_bias, self.limits,
            ],
            template=[
                ("T", proj.dtype),
                ("TT", tokens),
                ("INNER", self.inner),
                ("CONVD", self.conv_dim),
                ("HEADS", self.heads),
                ("DH", self.head_dim),
                ("DS", self.state_size),
                ("HPG", self.heads // self.groups),
                ("GROUPS", self.groups),
                ("K", self.kernel),
                ("HAS_CBIAS", self.has_bias),
            ],
            grid=(256, 1, self.heads),
            threadgroup=(256, 1, 1),
            output_shapes=[(tokens, self.inner), (tokens, *state.shape)],
            output_dtypes=[proj.dtype, state.dtype],
        )
        if gated_group_norm_applies(self.inner, self.groups):
            normed_rows = gated_group_norm(
                proj, out, self.norm_weight, self.eps_array,
                inner=self.inner, groups=self.groups, rows=tokens,
                proj_stride=proj.shape[-1],
            )
            return normed_rows, slots
        gate = proj[:, : self.inner]
        gated = (gate * mx.sigmoid(gate) * out).astype(mx.float32)
        grouped = mx.unflatten(gated, axis=-1, shape=(self.groups, -1))
        normed = mx.fast.rms_norm(grouped, None, self.eps).flatten(-2)
        return self.norm_weight * normed.astype(proj.dtype), slots
