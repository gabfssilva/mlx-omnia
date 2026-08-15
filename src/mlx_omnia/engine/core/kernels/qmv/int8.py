"""One-row activation through an affine INT8 group-32 stack in one dispatch.

Same meaning as `mx.quantized_matmul(..., transpose=True, group_size=32,
bits=8)` at T=1, with the decode-path optimizations that the mlxfast-challenge
(Layr Labs, MIT) promoted onto mlx's own quantized kernels. Baseline
`38fcfff9c865ff45244cb491629108d913e35d77` -> record
`c5b0a13c5cc032b485022db41bcd745792316714`; the geometry below is mlx 0.32's
`qmv_fast_impl` (`quantized.h`), which the record tree also transcribes
verbatim for its affine INT8 projection.

Carried from that tree:

- **uchar4 code load** (`quantized.h`, the record's only affine-INT8 hunk).
  Stock walks the eight codes a byte at a time; two 4-byte loads feed the same
  sequential accumulation chain, so the value is bit-identical.
- **vec<T,4> activation load** (`fp_quantized.h` at the challenge baseline).
  The same x elements land in the same `x_thread` slots with the same
  per-element T -> float conversion, and the running `sum` the affine bias term
  needs is accumulated in the same index order.
- **`unroll_count(4)` on the K contraction** (PRs 219/226). Scheduling only:
  the compiler pipelines the next block's code/scale/bias loads against the
  current block's dot instead of hoisting every block's loads up front. They
  report it as bit-exact and a decode win at a 2048-wide contraction.
- **Accumulator seed elision** (PRs 820/1241). The first product assigns the
  dot accumulator instead of being added to a dead `+0.0f`. Differs from stock
  only when that product is `-0.0f`, and `result[row] += scale * accum` absorbs
  the sign of zero.

Not carried, and why: the `2^14` renormalization fold and the split-nibble
`half2` spread are fp4 decode tricks -- an INT8 code is already an integer and
there is no fixed power-of-two to hoist. Scale-defer likewise needs a constant
factor shared by every group; affine scales vary per group and the `sum * bias`
term does not distribute out of the reduction at all. The register-prefetch
("A-operand-hoist") variant exists to put the weight stream in flight while a
fused RMS prologue runs dead-serial ahead of it; a bare projection has no such
prologue for it to hide behind.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.qmv.kernel import Epilogue, QmvLeaf, QmvStrategy
from mlx_omnia.engine.core.mxcompat import metal_kernel
from mlx_omnia.engine.core.patch import Patch

_HEADER = """
    // Eight codes as two 4-byte loads. Same operand order and same sequential
    // accumulation chain as the byte-at-a-time loop, so the sum is unchanged;
    // the first product assigns rather than seeding off a dead +0.0f.
    inline float qint8Dot(const device uchar4* ws, const thread float* x) {
        float accum;
        {
            uchar4 codes = ws[0];
            accum  = x[0] * codes[0];
            accum += x[1] * codes[1];
            accum += x[2] * codes[2];
            accum += x[3] * codes[3];
        }
        {
            uchar4 codes = ws[1];
            accum += x[4] * codes[0];
            accum += x[5] * codes[1];
            accum += x[6] * codes[2];
            accum += x[7] * codes[3];
        }
        return accum;
    }
"""

_SOURCE = """
    constexpr uint values_per_thread = 8;
    constexpr uint block_size = values_per_thread * 32;
    constexpr uint num_simdgroups = 2;
    constexpr uint group_size = 32;
    constexpr uint scale_step_per_thread = group_size / values_per_thread;
    constexpr uint kdim = (uint)KD;
    constexpr uint kgroups = kdim / group_size;

    uint tile = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;

    uint out_row = tile * (num_simdgroups * (uint)ROWS) + sg * (uint)ROWS;

    const device uint8_t* ws = (const device uint8_t*)W
        + (size_t)out_row * kdim + lane * values_per_thread;
    const device T* sl = S + (size_t)out_row * kgroups + lane / scale_step_per_thread;
    const device T* bl = Bs + (size_t)out_row * kgroups + lane / scale_step_per_thread;
    const device T* xv = X + lane * values_per_thread;

    float result[ROWS];
    for (uint row = 0; row < (uint)ROWS; row++) {
        result[row] = 0.0f;
    }

    #pragma clang loop unroll_count(4)
    for (uint k = 0; k < kdim; k += block_size) {
        float xt[values_per_thread];
        float xsum = 0.0f;
        const device metal::vec<T, 4>* xq = (const device metal::vec<T, 4>*)xv;
        for (uint p = 0; p < values_per_thread / 4; p++) {
            metal::vec<T, 4> q = xq[p];
            xt[p * 4 + 0] = (float)q.x; xsum += xt[p * 4 + 0];
            xt[p * 4 + 1] = (float)q.y; xsum += xt[p * 4 + 1];
            xt[p * 4 + 2] = (float)q.z; xsum += xt[p * 4 + 2];
            xt[p * 4 + 3] = (float)q.w; xsum += xt[p * 4 + 3];
        }
        for (uint row = 0; row < (uint)ROWS; row++) {
            float s = (float)sl[row * kgroups];
            float b = (float)bl[row * kgroups];
            float d = qint8Dot((const device uchar4*)(ws + row * kdim), xt);
            result[row] += s * d + xsum * b;
        }
        ws += block_size;
        sl += block_size / group_size;
        bl += block_size / group_size;
        xv += block_size;
    }

    for (uint row = 0; row < (uint)ROWS; row++) {
        float t = simd_sum(result[row]);
        if (lane == 0) {
            Y[out_row + row] = (T)t;
        }
    }
"""

_KERNEL = metal_kernel(
    name="int8_qmv",
    input_names=["X", "W", "S", "Bs"],
    output_names=["Y"],
    source=_SOURCE,
    header=_HEADER,
)

_GATED_SOURCE = """
    constexpr uint in_vec_size = (uint)KDIM;
    constexpr uint out_vec_size = (uint)ROWS;
    constexpr uint gate_heads = (uint)HEADS;
    constexpr uint head_shift = 7;
    constexpr uint values_per_thread = 8;
    constexpr uint block_size = 256;
    constexpr uint results_per_simdgroup = 4;
    constexpr uint num_simdgroups = 2;
    constexpr uint group_size = 32;
    constexpr uint scale_step_per_thread = group_size / values_per_thread;
    constexpr uint in_vec_size_g = in_vec_size / group_size;

    uint tile = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    threadgroup float gate_table[gate_heads];
    if (lid < gate_heads) {
        float gate = float(G[lid]);
        if (!(bool)PREACT) {
            if (metal::isnan(gate)) {
                gate = NAN;
            } else {
                float maxval = metal::max(gate, 0.0f);
                float minval = metal::min(gate, 0.0f);
                gate = (metal::isinf(minval) || metal::isinf(maxval))
                    ? maxval
                    : maxval + log1p(metal::exp(minval - maxval));
            }
        }
        gate_table[lid] = float(bfloat(gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint out_row = tile * (num_simdgroups * results_per_simdgroup)
        + simd_gid * results_per_simdgroup;
    const device uint8_t* ws = (const device uint8_t*)W
        + out_row * in_vec_size + simd_lid * values_per_thread;
    const device bfloat* sc = S + out_row * in_vec_size_g
        + simd_lid / scale_step_per_thread;
    const device bfloat* bs = Bs + out_row * in_vec_size_g
        + simd_lid / scale_step_per_thread;
    const device bfloat* xp = X + simd_lid * values_per_thread;

    float x_thread[values_per_thread];
    float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint column = simd_lid * values_per_thread;
    for (uint k = 0; k < in_vec_size; k += block_size) {
        float gate = gate_table[column >> head_shift];
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            float value = float(bfloat(float(xp[i]) * gate));
            sum += value;
            x_thread[i] = value;
        }
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + row * in_vec_size;
            float scale = float(sc[row * in_vec_size_g]);
            float bias = float(bs[row * in_vec_size_g]);
            float accum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                accum += x_thread[i] * wl[i];
            }
            result[row] += scale * accum + sum * bias;
        }
        ws += block_size;
        sc += block_size / group_size;
        bs += block_size / group_size;
        xp += block_size;
        column += block_size;
    }
    for (uint row = 0; row < results_per_simdgroup; ++row) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
            Y[out_row + row] = bfloat(result[row]);
        }
    }
"""

_GATED_KERNEL = metal_kernel(
    name="gated_int8_qmv",
    input_names=["X", "G", "W", "S", "Bs"],
    output_names=["Y"],
    source=_GATED_SOURCE,
)

_NORM_SOURCE = """
    constexpr uint axis_size = (uint)KDIM;
    constexpr uint n_reads = 4;
    constexpr uint norm_threads = axis_size / n_reads;
    constexpr uint real_threads = 64;
    constexpr uint virtual_per_thread = norm_threads / real_threads;
    constexpr uint simd_size = 32;
    constexpr float norm_eps = 1.0e-6f;
    constexpr uint values_per_thread = 8;
    constexpr uint block_size = 256;
    constexpr uint results_per_simdgroup = 4;
    constexpr uint num_simdgroups = 2;
    constexpr uint group_size = 32;
    constexpr uint scale_step_per_thread = group_size / values_per_thread;
    constexpr uint in_vec_size_g = axis_size / group_size;
    constexpr uint pf_depth = 4;

    uint tile = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[simd_size];
    uint out_row = tile * (num_simdgroups * results_per_simdgroup)
        + simd_gid * results_per_simdgroup;
    const device uint8_t* ws = (const device uint8_t*)W
        + out_row * axis_size + simd_lid * values_per_thread;
    const device bfloat* sc = S + out_row * in_vec_size_g
        + simd_lid / scale_step_per_thread;
    const device bfloat* bs = Bs + out_row * in_vec_size_g
        + simd_lid / scale_step_per_thread;
    uint8_t pf_w[pf_depth][results_per_simdgroup][values_per_thread];
    float pf_s[pf_depth][results_per_simdgroup];
    float pf_b[pf_depth][results_per_simdgroup];
    for (uint d = 0; d < pf_depth; ++d) {
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + d * block_size + row * axis_size;
            for (uint i = 0; i < values_per_thread; ++i) {
                pf_w[d][row][i] = wl[i];
            }
            pf_s[d][row] = float(sc[d * (block_size / group_size)
                + row * in_vec_size_g]);
            pf_b[d][row] = float(bs[d * (block_size / group_size)
                + row * in_vec_size_g]);
        }
    }
    if (lid < simd_size) {
        local_sums[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < virtual_per_thread; ++j) {
        uint base = (lid + j * real_threads) * n_reads;
        float acc = 0.0f;
        for (uint i = 0; i < n_reads; ++i) {
            float xi = float(X[base + i]);
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (simd_lid == 0) {
            local_sums[simd_gid + num_simdgroups * j] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float total = simd_sum(local_sums[simd_lid]);
        if (simd_lid == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(total / float(axis_size) + norm_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_mean = local_inv_mean[0];
    float x_thread[values_per_thread];
    float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint column = simd_lid * values_per_thread;
    for (uint d = 0; d < pf_depth; ++d) {
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            float value = float(bfloat(N[column + i]
                * bfloat(float(X[column + i]) * inv_mean)));
            sum += value;
            x_thread[i] = value;
        }
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            float accum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                accum += x_thread[i] * pf_w[d][row][i];
            }
            result[row] += pf_s[d][row] * accum + sum * pf_b[d][row];
        }
        ws += block_size;
        sc += block_size / group_size;
        bs += block_size / group_size;
        column += block_size;
    }
    for (uint k = pf_depth * block_size; k < axis_size; k += block_size) {
        float sum = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
            float value = float(bfloat(N[column + i]
                * bfloat(float(X[column + i]) * inv_mean)));
            sum += value;
            x_thread[i] = value;
        }
        for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device uint8_t* wl = ws + row * axis_size;
            float scale = float(sc[row * in_vec_size_g]);
            float bias = float(bs[row * in_vec_size_g]);
            float accum = 0.0f;
            for (uint i = 0; i < values_per_thread; ++i) {
                accum += x_thread[i] * wl[i];
            }
            result[row] += scale * accum + sum * bias;
        }
        ws += block_size;
        sc += block_size / group_size;
        bs += block_size / group_size;
        column += block_size;
    }
    for (uint row = 0; row < results_per_simdgroup; ++row) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
            Y[out_row + row] = bfloat(result[row]);
        }
    }
"""

_NORM_KERNEL = metal_kernel(
    name="norm_int8_qmv",
    input_names=["X", "N", "W", "S", "Bs"],
    output_names=["Y"],
    source=_NORM_SOURCE,
)


def norm_int8_qmv(
    x: mx.array,
    norm_weight: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    kdim = x.size
    rows = weight.shape[-2]
    assert x.dtype == mx.bfloat16 and norm_weight.dtype == mx.bfloat16
    assert int8_qmv_applies(kdim, rows, group_size=32, bits=8, rows_per_simd=4)
    assert norm_weight.shape == (kdim,)
    assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16
    assert scales.shape == (rows, kdim // 32) and biases.shape == scales.shape
    return _NORM_KERNEL(
        inputs=[x, norm_weight, weight, scales, biases],
        template=[("KDIM", kdim)],
        grid=((rows // 8) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.bfloat16],
    )[0]


def gated_int8_qmv_applies(
    attention: mx.array,
    gate_logits: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> bool:
    kdim = attention.size
    rows = weight.shape[-2]
    heads = gate_logits.size
    return (
        attention.dtype == mx.bfloat16
        and gate_logits.dtype == mx.bfloat16
        and scales.dtype == mx.bfloat16
        and biases.dtype == mx.bfloat16
        and kdim == heads * 128
        and kdim % 256 == 0
        and rows % 8 == 0
        and weight.shape[-1] * 4 == kdim
        and scales.shape == (rows, kdim // 32)
        and biases.shape == scales.shape
    )


def gated_int8_qmv(
    attention: mx.array,
    gate_logits: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    preactivated: bool = False,
) -> mx.array:
    assert gated_int8_qmv_applies(attention, gate_logits, weight, scales, biases)
    rows = weight.shape[-2]
    return _GATED_KERNEL(
        inputs=[attention, gate_logits, weight, scales, biases],
        template=[
            ("KDIM", attention.size),
            ("ROWS", rows),
            ("HEADS", gate_logits.size),
            ("PREACT", preactivated),
        ],
        grid=((rows // 8) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*attention.shape[:-1], rows)],
        output_dtypes=[mx.bfloat16],
    )[0]


def int8_qmv_applies(
    kdim: int, rows: int, *, group_size: int, bits: int, rows_per_simd: int
) -> bool:
    """The affine byte-per-code contraction has to tile with nothing bounds-checked.

    `bits`/`group_size` are pinned to 8/32: one code per byte is what the uchar4
    load reads, and four lanes per quantization group is what the scale index
    assumes. A simdgroup consumes 256 values per step (8 per lane) and a
    threadgroup owns `2 * rows_per_simd` output rows, so both dimensions have to
    divide exactly or the tail runs off the end of the buffers.
    """
    return bits == 8 and group_size == 32 and kdim % 256 == 0 and rows % (2 * rows_per_simd) == 0


def int8_qmv(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    rows_per_simd: int = 4,
) -> mx.array:
    """x [..., kdim] (one row) against a [rows, kdim] affine INT8 stack -> [..., rows].

    `rows_per_simd` is the one geometry knob: 4 is mlx's `qmv_fast` tiling and
    the tiling the record tree ships for this format; 1 is the one-row-per-SIMD
    retile (PRs 294/306/329), which trades four independent weight streams per
    simdgroup for four times as many threadgroups and one live accumulator.
    Every value is computed the same way either way.
    """
    kdim = weight.shape[-1] * 4
    rows = weight.shape[-2]
    assert x.size == kdim
    assert int8_qmv_applies(kdim, rows, group_size=32, bits=8, rows_per_simd=rows_per_simd)
    assert scales.shape[-1] == kdim // 32 and biases.shape == scales.shape
    return _KERNEL(
        inputs=[x, weight, scales, biases],
        template=[("T", x.dtype), ("KD", kdim), ("ROWS", rows_per_simd)],
        grid=(64, rows // (2 * rows_per_simd), 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[x.dtype],
    )[0]


def _stands_in_for_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    /,
    scales: mx.array | None = None,
    biases: mx.array | None = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
) -> bool:
    """Everything outside the one-row affine INT8 case stays on the stock op."""
    if not transpose or mode != "affine" or biases is None or scales is None:
        return False
    kdim = weight.shape[-1] * 4
    return x.size == kdim and int8_qmv_applies(
        kdim, weight.shape[-2], group_size=group_size, bits=bits, rows_per_simd=4
    )


def _as_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    /,
    scales: mx.array | None = None,
    biases: mx.array | None = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
) -> mx.array:
    assert scales is not None and biases is not None
    return int8_qmv(x, weight, scales, biases)


QUANTIZED_MATMUL = Patch(
    module=mx,
    name="quantized_matmul",
    applies=_stands_in_for_quantized_matmul,
    replacement=_as_quantized_matmul,
)


def gated_int8_qmv_shape_applies(
    kdim: int, rows: int, heads: int, *, group_size: int, bits: int
) -> bool:
    """`gated_int8_qmv_applies` over shapes alone, for a caller that resolves before it
    has the activation: the gate index is `column >> 7`, so a head is exactly 128 wide."""
    return (
        bits == 8 and group_size == 32 and kdim == heads * 128 and kdim % 256 == 0 and rows % 8 == 0
    )


@dataclass(frozen=True)
class Int8Qmv(QmvStrategy):
    weight: mx.array
    scales: mx.array
    biases: mx.array
    heads: int | None

    @classmethod
    def build(
        cls,
        leaf: QmvLeaf,
        *,
        kdim: int,
        rows: int,
        epilogue: Epilogue,
        heads: int | None,
    ) -> Self | None:
        if epilogue not in ("none", "gate") or not isinstance(leaf, nn.QuantizedLinear):
            return None
        if "bias" in leaf or leaf.mode != "affine" or leaf.biases is None:
            return None
        if epilogue == "gate":
            if heads is None or leaf.scales.dtype != mx.bfloat16:
                return None
            if leaf.biases.dtype != mx.bfloat16:
                return None
            if not gated_int8_qmv_shape_applies(
                kdim, rows, heads, group_size=leaf.group_size, bits=leaf.bits
            ):
                return None
            return cls(leaf.weight, leaf.scales, leaf.biases, heads)
        if not int8_qmv_applies(
            kdim, rows, group_size=leaf.group_size, bits=leaf.bits, rows_per_simd=4
        ):
            return None
        return cls(leaf.weight, leaf.scales, leaf.biases, None)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        if self.heads is None:
            return int8_qmv(x, self.weight, self.scales, self.biases)
        assert gate is not None
        return gated_int8_qmv(x, gate, self.weight, self.scales, self.biases, preactivated=True)
