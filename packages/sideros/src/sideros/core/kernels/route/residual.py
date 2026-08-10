"""Residual add + rms_norm fused with the router gemv and its ordinal sort keys.

The decode-shaped half of the residual join: one token, and the router gemv folded into
the same dispatch that produced the vector it reads. The normed row is left in
threadgroup memory, so `experts` dot products of length `hidden` are served from there
instead of from a second pass over device memory, and the router logits, their sigmoid
scores and the ordinal sort keys (`route.ordinal`'s) all leave the kernel with the
residual stream — the selection that follows has nothing left to compute but the sort.
The plain pair without the router lives in `core.kernels.add_norm`; this module reuses
its tiling predicate and emits the same normalization tail.

The token is replicated across `experts / rows_per_group` threadgroups: every tile
recomputes the whole norm (only tile 0 writes the residual outputs) and then owns
`rows_per_group` router rows. That is a bandwidth-latency knob, not a numerical one, and
it changes the emitted source in three coupled ways, exactly as the reference did:

* with fewer rows than simdgroups, `rows_per_thread` bottoms out at 1 and the surplus
  simdgroups sit out the router phase behind `active_simd_groups`. The guard opens after
  the norm's barrier and closes after the logit write, so no thread is skipped past a
  barrier and no row goes unwritten. At `active_simd_groups == simd_groups` the guard is
  not emitted at all;
* at `rows_per_thread == 1` the block loop is unrolled four deep — **loads only**.
  `router_result[0]` stays one accumulator stepped in strict `(block, i)` order; giving
  each unrolled step its own partial would regroup the fp32 adds into a tree and lose
  bit-exactness. Retiling alone cannot add an outstanding load (`tiles * rows_per_group`
  is `experts` at every setting), so hoisting four blocks' weight loads is what raises
  the in-flight bytes;
* at `rows_per_thread > 1` the unroll is over rows instead, and the normed coefficients
  are staged in registers because every row reuses them.

The 4-element read and the resulting threadgroup width (`hidden / 4`) are not knobs:
each thread squares its own contiguous four elements, and moving either regroups the
fp32 sum and forfeits agreement with a single-row rms_norm.
"""

from string import Template
from typing import TYPE_CHECKING

import mlx.core as mx

from sideros.core.kernels.add_norm.rows import applies as residual_rms_norm_applies
from sideros.core.kernels.route.ordinal import ORDINAL_HEADER
from sideros.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from sideros.core.mxcompat import MetalKernel as RawMetalKernel

_N_READS = 4
_SIMD_SIZE = 32
_BLOCK_WIDTH = _SIMD_SIZE * _N_READS

_NORM_TAIL = """if (simd_group == 0) {
            local_sums[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            acc = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(acc / (float)axis_size + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_mean = local_inv_mean[0];"""

_UNROLLED_ACCUMULATE = """        uint column = simd_lane * n_reads;
        for (uint block = 0; block < router_blocks; block += 4) {
            vec<T, 4> rw[4];
            for (uint u = 0; u < 4; ++u) {
                const device vec<T, 4>* row_values =
                    (const device vec<T, 4>*)(
                        router_weight + router_row * axis_size +
                            column + u * block_width);
                rw[u] = row_values[0];
            }
            for (uint u = 0; u < 4; ++u) {
                uint column_u = column + u * block_width;
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[0] += float(rw[u][i]) *
                        float(normalized_row[column_u + i]);
                }
            }
            column += 4 * block_width;
        }"""

_ROWWISE_ACCUMULATE = """        thread float router_input[n_reads];

        uint column = simd_lane * n_reads;
        for (uint block = 0; block < router_blocks; ++block) {
            for (uint i = 0; i < n_reads; ++i) {
                router_input[i] = float(normalized_row[column + i]);
            }
            for (uint r = 0; r < rows_per_thread; ++r) {
                const device vec<T, 4>* row_values =
                    (const device vec<T, 4>*)(
                        router_weight + (router_row + r) * axis_size +
                            column);
                const vec<T, 4> rw = row_values[0];
                for (uint i = 0; i < n_reads; ++i) {
                    router_result[r] += float(rw[i]) * router_input[i];
                }
            }
            column += block_width;
        }"""

_ROUTER_STORE = """        T logit = T(router_result[r]);
        router_logits[router_row + r] = logit;
        float x = float(logit);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float score = x < 0.0f ? y : 1.0f - y;
        router_keys[router_row + r] = router_key_ordinal(
            -(score + float(correction_bias[router_row + r])));"""

_ROUTER_SOURCE = Template("""
constexpr uint axis_size = $axis_size;
constexpr uint n_reads = 4;
constexpr uint simd_size = 32;
constexpr uint rows_per_group = $rows_per_group;
constexpr uint rows_per_thread = $rows_per_thread;
constexpr uint active_simd_groups = $active_simd_groups;
constexpr uint block_width = 128;
constexpr uint router_blocks = axis_size / block_width;

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint base = lid * n_reads;

threadgroup float local_inv_mean[1];
threadgroup float local_sums[simd_size];
threadgroup T normalized_row[axis_size];

thread T values[n_reads];
float acc = 0.0f;
for (uint i = 0; i < n_reads; ++i) {
    T value = T(residual[base + i] + branch[base + i]);
    values[i] = value;
    if (tile == 0) {
        summed[base + i] = value;
    }
    float fv = float(value);
    acc += fv * fv;
}

acc = simd_sum(acc);
$norm_tail

for (uint i = 0; i < n_reads; ++i) {
    T value =
        weight[base + i] *
        T(float(values[i]) * inv_mean);
    normalized_row[base + i] = value;
    if (tile == 0) {
        normalized[base + i] = value;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

${guard_open}\
uint router_row = tile * rows_per_group + simd_group * rows_per_thread;
thread float router_result[rows_per_thread] = {$zeros};
$accumulate

for (uint r = 0; r < rows_per_thread; ++r) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        router_result[r] +=
            metal::simd_shuffle_down(router_result[r], delta);
    }
}
if (simd_lane == 0) {
    for (uint r = 0; r < rows_per_thread; ++r) {
        $router_store
    }
}
$guard_close
""")

def _tiling(hidden: int, rows_per_group: int) -> tuple[int, int, int]:
    """`(rows_per_thread, active_simd_groups, simd_groups)` for one tile of the fusion."""
    simd_groups = hidden // _N_READS // _SIMD_SIZE
    rows_per_thread = rows_per_group // simd_groups if rows_per_group >= simd_groups else 1
    return rows_per_thread, rows_per_group // rows_per_thread, simd_groups


def _router_source(hidden: int, rows_per_group: int) -> str:
    rows_per_thread, active_simd_groups, simd_groups = _tiling(hidden, rows_per_group)
    guarded = active_simd_groups < simd_groups
    return _ROUTER_SOURCE.substitute(
        axis_size=hidden,
        rows_per_group=rows_per_group,
        rows_per_thread=rows_per_thread,
        active_simd_groups=active_simd_groups,
        norm_tail=_NORM_TAIL,
        guard_open="        if (simd_group < active_simd_groups) {\n" if guarded else "",
        guard_close="        }\n" if guarded else "",
        zeros=", ".join(["0.0f"] * rows_per_thread),
        accumulate=_UNROLLED_ACCUMULATE if rows_per_thread == 1 else _ROWWISE_ACCUMULATE,
        router_store=_ROUTER_STORE,
    )


_ROUTER_KERNELS: dict[tuple[int, int], "RawMetalKernel"] = {}


def _router_kernel(hidden: int, rows_per_group: int) -> "RawMetalKernel":
    """One compiled kernel per `(hidden, rows_per_group)`.

    mlx keys its JIT library cache by kernel name and clears the entry when a name's
    source changes, so both parameters — which are baked into the source, not passed as
    template arguments — have to be in the name or the variants would thrash one entry.
    """
    key = (hidden, rows_per_group)
    kernel = _ROUTER_KERNELS.get(key)
    if kernel is None:
        kernel = metal_kernel(
            name=f"residual_rms_router_h{hidden}_rpg{rows_per_group}",
            input_names=["residual", "branch", "weight", "router_weight", "correction_bias", "eps"],
            output_names=["summed", "normalized", "router_logits", "router_keys"],
            source=_router_source(hidden, rows_per_group),
            header=ORDINAL_HEADER,
        )
        _ROUTER_KERNELS[key] = kernel
    return kernel


def residual_rms_router_applies(hidden: int, experts: int, rows_per_group: int) -> bool:
    """The norm's tiling, plus the router phase's.

    `rows_per_group` router rows per tile means `experts / rows_per_group` tiles, so it
    has to divide `experts` exactly — no partial tile is dispatched and no row is
    computed twice or missed. Above one row per thread the rows are dealt out one per
    simdgroup, so `rows_per_group` has to be a whole multiple of them; at one row per
    thread the block loop is unrolled four deep with no tail, which needs
    `hidden / 128` to be a multiple of 4.
    """
    if not residual_rms_norm_applies(hidden) or rows_per_group <= 0:
        return False
    if experts % rows_per_group != 0:
        return False
    rows_per_thread, _, simd_groups = _tiling(hidden, rows_per_group)
    if rows_per_group >= simd_groups and rows_per_group % simd_groups != 0:
        return False
    return rows_per_thread > 1 or (hidden // _BLOCK_WIDTH) % 4 == 0


def residual_rms_router(
    residual: mx.array,
    branch: mx.array,
    weight: mx.array,
    router_weight: mx.array,
    correction_bias: mx.array,
    *,
    eps: float,
    rows_per_group: int = 8,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """One token's residual join with the router gemv fused in.

    Returns `(residual + branch, its rms_norm, router logits, router sort keys)`. The
    keys are `router_ordinal`'s remapping of `-(sigmoid(logit) + correction_bias)`, so
    ascending unsigned order is descending routing score with ties to the lower index.

    `residual` and `branch` hold one token (`hidden` elements in any shape),
    `router_weight` is `(experts, hidden)` and `correction_bias` is `(experts,)` in fp32.
    """
    hidden = weight.size
    experts = correction_bias.size
    assert residual.size == hidden and branch.size == hidden
    assert residual.dtype == branch.dtype == weight.dtype == router_weight.dtype
    assert router_weight.shape == (experts, hidden)
    assert correction_bias.dtype == mx.float32
    assert residual_rms_router_applies(hidden, experts, rows_per_group)
    threads = hidden // _N_READS
    tiles = experts // rows_per_group
    logits_shape = (*residual.shape[:-1], experts)
    out = _router_kernel(hidden, rows_per_group)(
        inputs=[
            residual.reshape(hidden),
            branch.reshape(hidden),
            weight,
            router_weight,
            correction_bias,
            mx.array(eps, dtype=mx.float32),
        ],
        template=[("T", residual.dtype)],
        grid=(tiles * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(hidden,), (hidden,), (experts,), (experts,)],
        output_dtypes=[residual.dtype, residual.dtype, residual.dtype, mx.uint32],
    )
    return (
        out[0].reshape(residual.shape),
        out[1].reshape(residual.shape),
        out[2].reshape(logits_shape),
        out[3].reshape(logits_shape),
    )
