"""mHC junction — rms fold + sigmoid gates -> comb softmax -> Sinkhorn -> collapse — and
the matching re-expansion, one dispatch each.

The Sinkhorn is a serial chain of tiny reductions over an `hc x hc` matrix, and a
hyper-connected trunk runs two junctions per layer: on DeepSeek-V4-Flash that is 86
junctions per token, all dispatch latency. In the kernel the matrix lives in one float4
per lane of simdgroup 0 — row norms are lane-local sums, column norms one `simd_sum` per
column — and the 4->1 collapse is the whole threadgroup over `vec<T,4>` loads.

The junction takes the *raw* gemv output: `rms_norm(y, None, eps) @ fn.T` equals
`(y @ fn.T) * rsqrt(mean(y*y) + eps)`, so the gemv runs on the unnormalized input (the
matmul promotes bf16 x f32 to one fp32 dispatch) and the kernel recovers the inverse rms
from `x` itself — the whole threadgroup accumulates the sum of squares before simdgroup 0
scales the mixes. The `[mix, hc*hidden]` gemv stays outside: a single threadgroup cannot
read its 1.5 MB at full bandwidth. On the way out the collapse is rounded to T, its own
sum of squares reduced, and the sublayer's *weighted* rms_norm applied in place — the
collapsed copies only ever feed that norm, so the kernel returns the sublayer's input.

Idle lanes run branchless on clamped indices and are masked by `active = 0`; the
`1 - active` term in the softmax denominator only keeps their 0/0 from minting NaN.
The op order matches `HyperConnection`'s fallback exactly: `softmax(...) + eps` with
no eps in the denominator, then a column normalization, then `iters - 1` row/column
rounds, every normalization guarded by `+ eps`.

`hc_expand` is the junction's other half — `post * x` broadcast over the copies plus
`comb^T @ residual` — a pure map over the hidden axis that the op chain spends ~5
dispatches on. The comb weights are read per thread straight from the tiny fp32
tensors (small inputs land in `constant` memory: no address-spaced pointers to them).
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint lane = tid % 32;
    uint sg = tid / 32;

    constexpr float EPS = EPS_INT * 1e-9f;
    constexpr float NEPS = NEPS_INT * 1e-9f;
    constexpr uint MIX = (2 + HC) * HC;

    const device float* raw_rows = raw + row * NT * MIX;
    device float* post_out = post + row * HC;
    device float* comb_out = comb + row * HC * HC;

    threadgroup float raw_sum[MIX];
    if (tid < MIX) {
        float acc_raw = 0.0f;
        for (uint t = 0; t < (uint)NT; ++t) {
            acc_raw += raw_rows[t * MIX + tid];
        }
        raw_sum[tid] = acc_raw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* raw_row = raw_sum;

    threadgroup float pre_shared[HC];
    threadgroup float sg_ss[8];
    threadgroup float inv_rms_sh;

    using T4 = vec<T, 4>;
    const device T* x_row = x + row * (HC * D);
    const device T4* x_all = (const device T4*)x_row;

    float ss = 0.0f;
    for (uint i = tid; i < (uint)(HC * D / 4); i += 256) {
        float4 v = float4(x_all[i]);
        ss = metal::fma(v.x, v.x, ss);
        ss = metal::fma(v.y, v.y, ss);
        ss = metal::fma(v.z, v.z, ss);
        ss = metal::fma(v.w, v.w, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) sg_ss[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < 8; ++i) total += sg_ss[i];
        inv_rms_sh = metal::rsqrt(total / (float)(HC * D) + NEPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = inv_rms_sh;

    if (sg == 0) {
        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint llane = metal::min(lane, (uint)(HC - 1));

        float pre_z = raw_row[llane] * inv_rms * scale[0] + base[llane];
        float post_z = raw_row[HC + llane] * inv_rms * scale[1] + base[HC + llane];
        if (lane < (uint)HC) {
            pre_shared[lane] = 1.0f / (1.0f + metal::exp(-pre_z)) + EPS;
            post_out[lane] = 2.0f / (1.0f + metal::exp(-post_z));
        }

        float4 v = (*(const threadgroup float4*)(raw_row + 2 * HC + llane * HC)
                        * (inv_rms * scale[2])
                  + *(const device float4*)(base + 2 * HC + llane * HC)) * active;
        float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
        float4 e = metal::exp(v - row_max) * active;
        float4 r = e / (e.x + e.y + e.z + e.w + (1.0f - active)) + EPS * active;

        float4 col = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y),
                                    simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col;
        for (int iter = 1; iter < ITERS; ++iter) {
            r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
            col = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y),
                                 simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col;
        }
        if (lane < (uint)HC) {
            *(device float4*)(comb_out + lane * HC) = r;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float p0 = pre_shared[0];
    const float p1 = pre_shared[1];
    const float p2 = pre_shared[2];
    const float p3 = pre_shared[3];

    const device T4* x0 = (const device T4*)(x_row + 0 * D);
    const device T4* x1 = (const device T4*)(x_row + 1 * D);
    const device T4* x2 = (const device T4*)(x_row + 2 * D);
    const device T4* x3 = (const device T4*)(x_row + 3 * D);
    const device T4* nw4 = (const device T4*)nw;
    device T4* out4 = (device T4*)(normed + row * D);

    float4 keep[(D + 1023) / 1024];
    float css = 0.0f;
    uint n = 0;
    for (uint d = tid; d < (uint)D / 4; d += 256, ++n) {
        float4 acc = metal::fma(float4(p0), float4(x0[d]),
                     metal::fma(float4(p1), float4(x1[d]),
                     metal::fma(float4(p2), float4(x2[d]), float4(p3) * float4(x3[d]))));
        float4 r = float4(T4(acc));
        keep[n] = r;
        css = metal::fma(r.x, r.x, css);
        css = metal::fma(r.y, r.y, css);
        css = metal::fma(r.z, r.z, css);
        css = metal::fma(r.w, r.w, css);
    }
    css = simd_sum(css);
    if (lane == 0) sg_ss[sg] = css;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < 8; ++i) total += sg_ss[i];
        inv_rms_sh = metal::rsqrt(total / (float)D + NEPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv2 = inv_rms_sh;

    n = 0;
    for (uint d = tid; d < (uint)D / 4; d += 256, ++n) {
        out4[d] = T4(keep[n] * inv2 * float4(nw4[d]));
    }
"""

_KERNEL = metal_kernel(
    name="hc_junction",
    input_names=["x", "raw", "scale", "base", "nw"],
    output_names=["normed", "post", "comb"],
    source=_SOURCE,
    ensure_row_contiguous=True,
)

_EXPAND_PLAIN_SOURCE = """
    uint d = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (d >= (uint)(D / 4)) return;

    using T4 = vec<T, 4>;
    const device T4* xv = (const device T4*)(x + row * D);
    const device T4* rv = (const device T4*)(residual + row * HC * D);
    device T4* ov = (device T4*)(out + row * HC * D);

    float4 xd = float4(xv[d]);
    float4 r0 = float4(rv[0 * (D / 4) + d]);
    float4 r1 = float4(rv[1 * (D / 4) + d]);
    float4 r2 = float4(rv[2 * (D / 4) + d]);
    float4 r3 = float4(rv[3 * (D / 4) + d]);

    for (uint c = 0; c < (uint)HC; ++c) {
        float4 z = comb[row * HC * HC + 0 * HC + c] * r0;
        z = metal::fma(float4(comb[row * HC * HC + 1 * HC + c]), r1, z);
        z = metal::fma(float4(comb[row * HC * HC + 2 * HC + c]), r2, z);
        z = metal::fma(float4(comb[row * HC * HC + 3 * HC + c]), r3, z);
        ov[c * (D / 4) + d] = T4(metal::fma(float4(post[row * HC + c]), xd, z));
    }
"""

_EXPAND_PLAIN_KERNEL = metal_kernel(
    name="hc_expand",
    input_names=["x", "residual", "post", "comb"],
    output_names=["out"],
    source=_EXPAND_PLAIN_SOURCE,
    ensure_row_contiguous=True,
)

_EXPAND_EMIT_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint gx = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint tg = gx / 32;

    constexpr uint MIX = (2 + HC) * HC;

    using T4 = vec<T, 4>;
    const device T4* xv = (const device T4*)(x + row * D);
    const device T4* rv = (const device T4*)(residual + row * HC * D);
    const device float4* fn4 = (const device float4*)fn;
    device T4* ov = (device T4*)(out + row * HC * D);

    float acc[MIX];
    for (uint j = 0; j < MIX; ++j) acc[j] = 0.0f;

    for (uint d = gx; d < (uint)(D / 4); d += (uint)NT * 32) {
        float4 xd = float4(xv[d]);
        float4 r0 = float4(rv[0 * (D / 4) + d]);
        float4 r1 = float4(rv[1 * (D / 4) + d]);
        float4 r2 = float4(rv[2 * (D / 4) + d]);
        float4 r3 = float4(rv[3 * (D / 4) + d]);

        for (uint c = 0; c < (uint)HC; ++c) {
            float4 z = comb[row * HC * HC + 0 * HC + c] * r0;
            z = metal::fma(float4(comb[row * HC * HC + 1 * HC + c]), r1, z);
            z = metal::fma(float4(comb[row * HC * HC + 2 * HC + c]), r2, z);
            z = metal::fma(float4(comb[row * HC * HC + 3 * HC + c]), r3, z);
            T4 rounded = T4(metal::fma(float4(post[row * HC + c]), xd, z));
            ov[c * (D / 4) + d] = rounded;
            float4 e = float4(rounded);
            for (uint j = 0; j < MIX; ++j) {
                acc[j] += metal::dot(e, fn4[(j * (uint)HC + c) * (uint)(D / 4) + d]);
            }
        }
    }

    for (uint j = 0; j < MIX; ++j) {
        float total = simd_sum(acc[j]);
        if (tid == 0) {
            partials[(row * (uint)NT + tg) * MIX + j] = total;
        }
    }
"""

_EXPAND_EMIT_KERNEL = metal_kernel(
    name="hc_expand_emit",
    input_names=["x", "residual", "post", "comb", "fn"],
    output_names=["out", "partials"],
    source=_EXPAND_EMIT_SOURCE,
    ensure_row_contiguous=True,
)

_THREADS = 256


def hc_junction_applies(hc_mult: int, hidden: int) -> bool:
    """One float4 per comb row and a four-way unrolled collapse: `hc_mult` is exactly 4;
    the collapse reads `vec<T, 4>`, so the hidden size is a multiple of 4."""
    return hc_mult == 4 and hidden % 4 == 0


def _tiles(hidden: int) -> int:
    return max(1, min(32, hidden // 128))


def hc_junction(
    x: mx.array,
    partials: mx.array,
    scale: mx.array,
    base: mx.array,
    norm_weight: mx.array,
    *,
    iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """One junction from the mixes gemv's partial sums — `(B, L, NT, mix)`, one row per
    reducing tile, a plain gemv passing `NT = 1` — summed in a fixed order in-kernel:
    the copies collapsed under the `pre` gate and passed through the sublayer's weighted
    rms_norm (the collapse only ever feeds that norm), plus the `post` gate and the
    doubly-stochastic `comb` that re-expand the sublayer's output."""
    batch, length, hc, hidden = x.shape
    assert hc_junction_applies(hc, hidden)
    out = _KERNEL(
        inputs=[x, partials, scale, base, norm_weight],
        template=[
            ("T", x.dtype),
            ("HC", hc),
            ("ITERS", iters),
            ("D", hidden),
            ("NT", partials.shape[2]),
            ("EPS_INT", round(eps / 1e-9)),
            ("NEPS_INT", round(norm_eps / 1e-9)),
        ],
        grid=(batch * length * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(batch, length, hidden), (batch, length, hc), (batch, length, hc, hc)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )
    return out[0], out[1], out[2]


def hc_expand(
    x: mx.array,
    residual: mx.array,
    post: mx.array,
    comb: mx.array,
    fn: mx.array | None = None,
) -> tuple[mx.array, mx.array | None]:
    """The sublayer output broadcast back over the copies, plus the copies mixed. With
    `fn` — the *next* junction's mixing weights — each tile also dots its slice of the
    expanded copies against every `fn` row, so the following junction starts from these
    partial sums instead of paying its own serial gemv dispatch."""
    batch, length, hc, hidden = residual.shape
    assert hc_junction_applies(hc, hidden)
    if fn is None:
        (out,) = _EXPAND_PLAIN_KERNEL(
            inputs=[x, residual, post, comb],
            template=[("T", x.dtype), ("HC", hc), ("D", hidden)],
            grid=(hidden // 4, batch * length, 1),
            threadgroup=(min(256, hidden // 4), 1, 1),
            output_shapes=[(batch, length, hc, hidden)],
            output_dtypes=[x.dtype],
        )
        return out, None
    tiles = _tiles(hidden)
    mix = (2 + hc) * hc
    out = _EXPAND_EMIT_KERNEL(
        inputs=[x, residual, post, comb, fn],
        template=[("T", x.dtype), ("HC", hc), ("D", hidden), ("NT", tiles)],
        grid=(tiles * 32, batch * length, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, length, hc, hidden), (batch, length, tiles, mix)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out[0], out[1]
