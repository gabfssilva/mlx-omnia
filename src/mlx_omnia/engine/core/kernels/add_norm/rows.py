"""Residual add + rms_norm over a block of rows, in one dispatch.

One threadgroup per row, four contiguous elements per thread, the sum of squares
reduced through `simd_sum` and a 32-slot threadgroup array. It overlaps `fused.py` in
purpose but not in arithmetic — it walks many rows per dispatch, reduces with
`metal::precise::rsqrt`, and rounds the normed value to `T` *before* the norm weight
multiplies it, where `fused.py` keeps the product in fp32 and rounds once. Neither
rounding is a refinement of the other; they are two different numerical paths and each
is pinned by its own test.

The 4-element read and the resulting threadgroup width (`hidden / 4`) are not knobs:
each thread squares its own contiguous four elements, and moving either regroups the
fp32 sum and forfeits agreement with a single-row rms_norm.
"""

from dataclasses import dataclass
from string import Template
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.mxcompat import metal_kernel

_N_READS = 4
_SIMD_SIZE = 32
_BLOCK_WIDTH = _SIMD_SIZE * _N_READS
_MAX_THREADGROUP = 1024

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

_NORM_SOURCE = Template("""
constexpr uint axis_size = AXIS;
constexpr uint n_reads = 4;
constexpr uint simd_size = 32;

uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint base = row * axis_size + lid * n_reads;

threadgroup float local_inv_mean[1];
threadgroup float local_sums[simd_size];

thread T values[n_reads];
float acc = 0.0f;
for (uint i = 0; i < n_reads; ++i) {
    T value = T(residual[base + i] + branch[base + i]);
    values[i] = value;
    summed[base + i] = value;
    float fv = float(value);
    acc += fv * fv;
}

acc = simd_sum(acc);
$norm_tail

for (uint i = 0; i < n_reads; ++i) {
    normalized[base + i] =
        weight[lid * n_reads + i] *
        T(float(values[i]) * inv_mean);
}
""").substitute(norm_tail=_NORM_TAIL)

_NORM_KERNEL = metal_kernel(
    name="residual_rms_norm_rows",
    input_names=["residual", "branch", "weight", "eps"],
    output_names=["summed", "normalized"],
    source=_NORM_SOURCE,
)


def applies(hidden: int) -> bool:
    """One threadgroup per row, four contiguous elements per thread: the `hidden / 4`
    threads have to be a whole number of simdgroups (so `hidden` is a multiple of 128)
    and fit the device's 1024-thread limit, which also keeps the 32-slot partial-sum
    array wide enough for one entry per simdgroup."""
    threads = hidden // _N_READS
    return (
        hidden % _BLOCK_WIDTH == 0
        and threads <= _MAX_THREADGROUP
        and threads // _SIMD_SIZE <= _SIMD_SIZE
    )


@dataclass(frozen=True)
class RowsAddRmsNorm:
    weight: mx.array
    eps: float

    @classmethod
    def build(cls, leaf: nn.RMSNorm, *, tokens: int | None) -> Self | None:
        if not applies(leaf.weight.size):
            return None
        return cls(leaf.weight, leaf.eps)

    def __call__(self, x: mx.array, projected: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.weight.size
        assert x.shape == projected.shape and x.shape[-1] == hidden
        assert x.dtype == projected.dtype == self.weight.dtype
        rows = x.size // hidden
        threads = hidden // _N_READS
        out = _NORM_KERNEL(
            inputs=[x, projected, self.weight, mx.array(self.eps, dtype=mx.float32)],
            template=[("T", x.dtype), ("AXIS", hidden)],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape, x.shape],
            output_dtypes=[x.dtype, x.dtype],
        )
        return out[0], out[1]
