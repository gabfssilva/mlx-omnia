"""The Mamba2 T=1 middle fused up to the norm: conv, dt and SSD step in one dispatch.

One threadgroup per head keeps the whole GPU busy (64 threadgroups against the 8 a
group-per-threadgroup shape leaves): each convolves its head's 64 hidden channels plus
its group's B and C slices — the group slices are recomputed by each of the group's
heads, a few hundred flops against a dispatch saved — computes its head's dt, and runs
the recurrence with the exact per-row arithmetic of `ssm/step.py` (a simdgroup owns a
row, a lane owns `Ds/32` contiguous state entries, the same `fast::exp`). The window
slides in the same dispatch, written once: hidden channels by their head, the group's
B/C channels by the group's first head.

The grouped gated RMSNorm cannot join this dispatch — its group spans eight heads —
but it no longer rides the op chain either: `norm.py` replicates the gate-SiLU,
`rms_single_row` reduction and weight in one kernel of its own, replacing three
serial hops with one.

Rounding follows the op chain exactly: the conv chain is `conv_step`'s (each tap
product, accumulation, bias and SiLU rounded to T), dt is mlx's own `logaddexp(x, 0)`
in fp32 with the clamp applied, and the recurrence reproduces `ssm_step` bit-for-bit.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

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
    constexpr uint dh = DH;
    constexpr uint ds = DS;
    constexpr uint hpg = HPG;
    constexpr uint k = K;
    constexpr uint npt = ds / 32;
    constexpr uint rows_per_sg = dh / 8;

    uint g = n / hpg;

    threadgroup float xc[dh];
    threadgroup float bc[ds];
    threadgroup float cc[ds];
    threadgroup float dtf[1];

    // Phase 0: the conv chain for this head's channels and its group's B/C, plus dt.
    {
    #pragma clang fp contract(off)
    for (uint j = tid; j < dh + 2 * ds + 1; j += 256) {
        uint c = 0xffffffffu;
        threadgroup float* slot = nullptr;
        bool writes = false;
        if (j < dh) {
            c = n * dh + j;
            slot = xc + j;
            writes = true;
        } else if (j < dh + ds) {
            c = inner + g * ds + (j - dh);
            slot = bc + (j - dh);
            writes = (n % hpg) == 0;
        } else if (j < dh + 2 * ds) {
            c = inner + GROUPS * ds + g * ds + (j - dh - ds);
            slot = cc + (j - dh - ds);
            writes = (n % hpg) == 0;
        } else {
            float v = (float)P[inner + conv + n] + DTB[n];
            float mx_ = metal::max(v, 0.0f);
            float mn_ = metal::min(v, 0.0f);
            float sp = mx_ + ::log1p(metal::exp(mn_ - mx_));
            dtf[0] = metal::min(metal::max(sp, LIMS[0]), LIMS[1]);
            continue;
        }
        float m = (float)(T)((float)WIN[c] * (float)TAPS[c * k + 0]);
        for (uint t = 1; t + 1 < k; t++) {
            float p = (float)(T)((float)WIN[t * conv + c] * (float)TAPS[c * k + t]);
            m = (float)(T)(m + p);
        }
        float last = (float)(T)((float)P[inner + c] * (float)TAPS[c * k + (k - 1)]);
        m = (float)(T)(m + last);
        if (HAS_CBIAS) {
            m = (float)(T)(m + (float)CB[c]);
        }
        T mt = (T)m;
        T y = (T)1 / ((T)1 + metal::precise::exp(metal::abs(mt)));
        T s = (mt < (T)0) ? y : (T)1 - y;
        *slot = (float)(T)((T)m * s);
        if (writes) {
            for (uint t = 0; t + 2 < k; t++) {
                WOUT[t * conv + c] = WIN[(t + 1) * conv + c];
            }
            WOUT[(k - 2) * conv + c] = P[inner + c];
        }
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: the SSD recurrence, `ssm_step`'s own per-row arithmetic.
    float dt_ = dtf[0];
    float A = -metal::fast::exp(AL[n]);
    float dA = metal::fast::exp(A * dt_);
    for (uint r = sg * rows_per_sg; r < (sg + 1) * rows_per_sg; r++) {
        float x_ = xc[r];
        size_t base = ((size_t)n * dh + r) * ds;
        float acc = 0.0f;
        for (uint i = 0; i < npt; ++i) {
            uint s_idx = npt * lane + i;
            float dB_by_x = x_ * dt_ * bc[s_idx];
            float st = dA * SIN[base + s_idx] + dB_by_x;
            SOUT[base + s_idx] = st;
            acc += st * cc[s_idx];
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            Y[n * dh + r] = (T)(acc + x_ * DD[n]);
        }
    }
"""

_KERNEL = metal_kernel(
    name="mamba_step_fused",
    input_names=["P", "WIN", "TAPS", "CB", "SIN", "AL", "DD", "DTB", "LIMS"],
    output_names=["Y", "WOUT", "SOUT"],
    source=_SOURCE,
)


@dataclass(frozen=True)
class FusedMambaStep:
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
    def build(
        cls,
        *,
        taps: mx.array,
        conv_bias: mx.array | None,
        A_log: mx.array,
        D: mx.array,
        dt_bias: mx.array,
        norm_weight: mx.array,
        eps: float,
        inner: int,
        conv_dim: int,
        kernel: int,
        heads: int,
        head_dim: int,
        groups: int,
        state_size: int,
        time_step_limit: tuple[float, float],
    ) -> Self | None:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
        if kernel < 2 or heads % groups != 0:
            return None
        if state_size % 32 != 0 or head_dim % 8 != 0:
            return None
        if heads * head_dim != inner:
            return None
        if conv_dim != inner + 2 * groups * state_size:
            return None
        filler = conv_bias if conv_bias is not None else mx.zeros((conv_dim,), taps.dtype)
        return cls(
            taps,
            filler,
            A_log.astype(mx.float32),
            D.astype(mx.float32),
            dt_bias.astype(mx.float32),
            norm_weight,
            mx.array(time_step_limit, dtype=mx.float32),
            mx.array([eps], dtype=mx.float32),
            eps,
            inner,
            conv_dim,
            kernel,
            heads,
            head_dim,
            groups,
            state_size,
            conv_bias is not None,
        )

    def __call__(
        self, proj: mx.array, window: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        out, slid, advanced = _KERNEL(
            inputs=[
                proj, window, self.taps, self.conv_bias, state,
                self.A_log, self.D, self.dt_bias, self.limits,
            ],
            template=[
                ("T", proj.dtype),
                ("INNER", self.inner),
                ("CONVD", self.conv_dim),
                ("DH", self.head_dim),
                ("DS", self.state_size),
                ("HPG", self.heads // self.groups),
                ("GROUPS", self.groups),
                ("K", self.kernel),
                ("HAS_CBIAS", self.has_bias),
            ],
            grid=(256, 1, self.heads),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (self.inner,),
                (self.kernel - 1, self.conv_dim),
                state.shape,
            ],
            output_dtypes=[proj.dtype, proj.dtype, state.dtype],
        )
        if gated_group_norm_applies(self.inner, self.groups):
            normed = gated_group_norm(
                proj, out, self.norm_weight, self.eps_array,
                inner=self.inner, groups=self.groups, rows=1, proj_stride=proj.size,
            )
            return normed, slid, advanced
        gate = proj[: self.inner]
        gated = (gate * mx.sigmoid(gate) * out).astype(mx.float32)
        grouped = mx.unflatten(gated, axis=-1, shape=(self.groups, -1))
        normed = mx.fast.rms_norm(grouped, None, self.eps).flatten(-2)
        return self.norm_weight * normed.astype(proj.dtype), slid, advanced
