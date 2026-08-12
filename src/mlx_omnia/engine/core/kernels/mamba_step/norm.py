"""The Mamba2 strategies' post-SSD tail in one dispatch: gate SiLU, grouped RMSNorm, weight.

The tail the op chain spells as

    gated = (gate * mx.sigmoid(gate) * out).astype(mx.float32)
    normed = mx.fast.rms_norm(mx.unflatten(gated, ...), None, eps)
    y = norm_weight * normed.flatten(...).astype(dtype)

is three serial hops per layer under the compiled graph. This kernel is a textual
replica of that chain: the SiLU rounds in T exactly as the fused step's own sigmoid
idiom, and the reduction is mlx's `rms_single_row` (`rms_norm.metal`) — per-thread
`N_READS = 4` sequential fp32 squares, `simd_sum`, one shared slot per simdgroup,
`simd_sum` again, `precise::rsqrt(acc / axis + eps)` — with the same 4-per-thread
partition, so the output is bit-identical to the ops it replaces.

One threadgroup per (row, group): `group_size / 4` threads, the exact shape mlx
launches for a 512-wide fp32 row.
"""

import mlx.core as mx

from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint group = threadgroup_position_in_grid.y;
    uint row = threadgroup_position_in_grid.z;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    constexpr uint gsize = GSIZE;
    constexpr uint inner = INNER;
    constexpr uint pstride = PSTRIDE;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];

    uint base = group * gsize + lid * 4;
    const device T* g = P + (size_t)row * pstride + base;
    const device T* o = O + (size_t)row * inner + base;

    // The op chain's own elementwise rounding: sigmoid in T via the mirrored stable
    // form, each product rounded to T before the fp32 cast.
    float gated[4];
    float acc = 0.0f;
    for (uint i = 0; i < 4; i++) {
        T gt = g[i];
        T y = (T)1 / ((T)1 + metal::precise::exp(metal::abs(gt)));
        T s = (gt < (T)0) ? y : (T)1 - y;
        T t1 = gt * s;
        T t2 = t1 * o[i];
        float xi = (float)t2;
        gated[i] = xi;
        acc += xi * xi;
    }

    // rms_single_row's reduction, verbatim.
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / gsize + EPS[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device T* out_ = Y + (size_t)row * inner + base;
    for (uint i = 0; i < 4; i++) {
        out_[i] = W[base + i] * (T)(gated[i] * local_inv_mean[0]);
    }
"""

_KERNEL = metal_kernel(
    name="mamba_gated_group_norm",
    input_names=["P", "O", "W", "EPS"],
    output_names=["Y"],
    source=_SOURCE,
)


def gated_group_norm_applies(inner: int, groups: int) -> bool:
    """One threadgroup per group row with four values per thread — the exact launch
    mlx's `rms_single_row` uses for the same width, which is what keeps the reduction
    order identical."""
    gsize = inner // groups
    return inner % groups == 0 and gsize % 128 == 0 and gsize // 4 <= 1024


def gated_group_norm(
    proj: mx.array,
    out: mx.array,
    weight: mx.array,
    eps_arr: mx.array,
    *,
    inner: int,
    groups: int,
    rows: int,
    proj_stride: int,
) -> mx.array:
    """`(rows, inner)` (or `(inner,)` when rows == 1) of the finished mixer output.

    `proj` supplies the gate — its first `inner` columns — so the caller passes the
    projection rows themselves and no slice is materialized."""
    gsize = inner // groups
    result = _KERNEL(
        inputs=[proj, out, weight, eps_arr],
        template=[
            ("T", out.dtype),
            ("GSIZE", gsize),
            ("INNER", inner),
            ("PSTRIDE", proj_stride),
        ],
        grid=(gsize // 4, groups, rows),
        threadgroup=(gsize // 4, 1, 1),
        output_shapes=[(rows, inner) if rows > 1 else (inner,)],
        output_dtypes=[out.dtype],
    )
    return result[0]
