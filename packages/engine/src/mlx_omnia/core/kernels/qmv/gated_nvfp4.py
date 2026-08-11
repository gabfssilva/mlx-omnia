"""Per-head gate product and a one-row NVFP4 projection in one dispatch.

Transcribed from the mlxfast-challenge (Layr Labs, MIT) record tree's
`laguna_oproj_act_*_v1_lm1_pw1_sc1_se1` -- the pre-activated-gate, lane-major, pairwise
arm of its output projection. Every line is the record's, down to its indentation; the
only edits are the four shape constants becoming template arguments. The op chain it
replaces is a broadcast multiply of the
attention output by one already-activated gate value per head, rounded to bf16, followed
by `mx.quantized_matmul(..., transpose=True, group_size=16, bits=4, mode="nvfp4")`. The
gate product happens at the same fp32 -> bf16 rounding point, so it is the same input the
two-dispatch chain contracts.

Its decode geometry is `qmv.nvfp4`'s -- split-nibble `half2` spread, `2^22` folded out of
the K loop and re-applied after `simd_sum`, seed elision, sign-domain scale decode; see
that module for why each is bit-exact -- retiled to four output rows per simdgroup, which
is what makes one gate load and one activation load serve four weight streams. The K
contraction is wide enough here for a lane's scale nibbles to span several bytes, so this
arm walks the run a nibble at a time (`nsh` toggling between the two halves of a byte)
instead of `qmv.nvfp4`'s single `ushort`; that is the only structural difference between
the two, and it is why this kernel is not pinned to a 2048-wide contraction.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.qmv.kernel import Epilogue, QmvLeaf
from mlx_omnia.core.kernels.shared.nvfp4 import LaneMajorScales, lane_major_scales
from mlx_omnia.core.mxcompat import metal_kernel

_BLOCK_SIZE = 512
_ROWS_PER_THREADGROUP = 8
_GROUP_SIZE = 16
_BITS = 4

_SOURCE = """
constexpr uint in_vec_size = (uint)IN_VEC;
constexpr uint out_vec_size = (uint)OUT_VEC;
constexpr uint gate_heads = (uint)HEADS;
constexpr uint head_shift = (uint)HEAD_SHIFT;
constexpr uint group_size = 16;
constexpr uint values_per_thread = 16;
constexpr uint codes_per_thread = values_per_thread / 8;
constexpr uint block_size = values_per_thread * 32;
constexpr uint results_per_simdgroup = 4;
constexpr uint num_simdgroups = 2;
constexpr uint in_vec_size_g = in_vec_size / group_size;

uint tile = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;



uint out_row = tile * (num_simdgroups * results_per_simdgroup) +
    simd_gid * results_per_simdgroup;
const device uint32_t* ws =
    (const device uint32_t*)weight_codes +
    out_row * (in_vec_size / 8) + simd_lid * codes_per_thread;
const device uint8_t* nq = scale_nibbles +
    out_row * (in_vec_size_g / 4) +
    (simd_lid >> 1) * (in_vec_size_g / 64);
const device uint8_t* bs = scale_bases + out_row;
const device uint8_t* sc = weight_scales +
    out_row * in_vec_size_g + simd_lid;
uint nsh = 0;
const device bfloat* xp = attention_output + simd_lid * values_per_thread;

thread float x_thread[values_per_thread];
thread float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < in_vec_size; k += block_size) {
    float g=float(gate_values[column>>head_shift]);
for(uint i=0;i<values_per_thread;++i)
    x_thread[i]=float(bfloat(float(xp[i])*g));

    for (uint row = 0; row < results_per_simdgroup; ++row) {
        const device uint32_t* wl = ws + row * (in_vec_size / 8);
        const uint8_t rb = bs[row];
    const bool esc = rb == 0xFFu;
    const device uint8_t* sp = esc
        ? (sc + row * in_vec_size_g)
        : (nq + row * (in_vec_size_g / 4));
    const uint8_t raw = sp[0];
    uint8_t sbits = esc ? raw : uint8_t(rb + ((raw >> nsh) & 0x0Fu));
        ushort sraw = ushort(sbits) << 7;
        float scale = float(as_type<half>(sraw));
        float accum;
        #pragma unroll
        for (uint j = 0; j < codes_per_thread; ++j) {
            const uint c = wl[j];
                            const uint xe = c & 0x0F0F0F0Fu;
                const uint ge = xe | (xe << 3);
                const uint yo = c & 0xF0F0F0F0u;
                const uint go = yo | (yo >> 3);
                const uint p0 = (ge << 9) & 0x8E008E00u;
                const uint p1 = (go << 8) & 0x8E008E00u;
                const uint p2 = (ge << 1) & 0x8E008E00u;
                const uint p3 = go & 0x8E008E00u;
            const float2 v04 = float2(as_type<half2>(p0));
            const float2 v15 = float2(as_type<half2>(p1));
            const float2 v26 = float2(as_type<half2>(p2));
            const float2 v37 = float2(as_type<half2>(p3));
            if (j == 0) {
                accum =
                    (x_thread[8 * j] * v04.x +
                     x_thread[8 * j + 1] * v15.x +
                     x_thread[8 * j + 2] * v26.x +
                     x_thread[8 * j + 3] * v37.x);
            } else {
                accum +=
                    (x_thread[8 * j] * v04.x +
                     x_thread[8 * j + 1] * v15.x +
                     x_thread[8 * j + 2] * v26.x +
                     x_thread[8 * j + 3] * v37.x);
            }
            accum +=
                (x_thread[8 * j + 4] * v04.y +
                 x_thread[8 * j + 5] * v15.y +
                 x_thread[8 * j + 6] * v26.y +
                 x_thread[8 * j + 7] * v37.y);
        }
        result[row] += scale * accum;
    }

    ws += block_size / 8;
    sc += block_size / group_size;
nq += nsh >> 2;
nsh ^= 4;
    xp += block_size;
    column += block_size;
}

for (uint row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row] * 4194304.0f);
    if (simd_lid == 0) {
        projected[out_row + row] = bfloat(result[row]);
    }
}
"""

_KERNEL = metal_kernel(
    name="gated_nvfp4_qmv_lane_major",
    input_names=[
        "attention_output",
        "gate_values",
        "weight_codes",
        "scale_nibbles",
        "scale_bases",
        "weight_scales",
    ],
    output_names=["projected"],
    source=_SOURCE,
    header="",
)


def gated_nvfp4_qmv_applies(
    kdim: int, rows: int, heads: int, *, group_size: int, bits: int
) -> bool:
    """The contraction, the output tiling and the gate's index shift all have to be exact.

    `bits`/`group_size` pinned to 4/16 is the `uint2`-per-lane group. A simdgroup steps
    512 values with nothing bounds-checked, and it walks its lane-major scale run one
    nibble per step alternating within a byte, so the run must be a whole number of bytes:
    64 groups, i.e. 1024 values, per lane pair's byte. Two simdgroups own four output rows
    each. The gate index is `column >> head_shift`, so the head width has to be a power of
    two that divides the contraction into exactly `heads` heads.
    """
    if bits != _BITS or group_size != _GROUP_SIZE:
        return False
    if kdim % (2 * _BLOCK_SIZE) != 0 or rows % _ROWS_PER_THREADGROUP != 0:
        return False
    if heads <= 0 or kdim % heads != 0:
        return False
    head_dim = kdim // heads
    return head_dim & (head_dim - 1) == 0


def gated_nvfp4_qmv(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    scales: mx.array,
    bank: LaneMajorScales,
) -> mx.array:
    """x [..., kdim] (one row, bf16) scaled per head by the already-activated `gate`
    [..., heads] (bf16), then projected through a [rows, kdim] NVFP4 stack -> [..., rows].

    `scales` is the stock `[rows, kdim / 16]` e4m3 plane and `bank` its lane-major
    packing; both stay bound because an escaped row reads the plane.
    """
    kdim = weight.shape[-1] * 8
    rows = weight.shape[-2]
    heads = gate.size
    assert x.dtype == mx.bfloat16 and gate.dtype == mx.bfloat16 and x.size == kdim
    assert gated_nvfp4_qmv_applies(kdim, rows, heads, group_size=_GROUP_SIZE, bits=_BITS)
    assert scales.shape == (rows, kdim // _GROUP_SIZE)
    assert bank.nibbles.shape == (rows, kdim // (4 * _GROUP_SIZE))
    assert bank.bases.shape == (rows,)
    return _KERNEL(
        inputs=[x, gate, weight, bank.nibbles, bank.bases, scales],
        template=[
            ("IN_VEC", kdim),
            ("OUT_VEC", rows),
            ("HEADS", heads),
            ("HEAD_SHIFT", (kdim // heads).bit_length() - 1),
        ],
        grid=((rows // _ROWS_PER_THREADGROUP) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.bfloat16],
    )[0]


@dataclass(frozen=True)
class GatedNvfp4Qmv:
    weight: mx.array
    scales: mx.array
    bank: LaneMajorScales
    heads: int

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
        if epilogue != "gate" or heads is None:
            return None
        if not isinstance(leaf, nn.QuantizedLinear) or "bias" in leaf:
            return None
        if (leaf.mode, leaf.group_size, leaf.bits) != ("nvfp4", _GROUP_SIZE, _BITS):
            return None
        if not gated_nvfp4_qmv_applies(
            kdim, rows, heads, group_size=leaf.group_size, bits=leaf.bits
        ):
            return None
        bank = lane_major_scales(leaf.scales)
        mx.eval(bank.nibbles, bank.bases)
        return cls(leaf.weight, leaf.scales, bank, heads)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        assert gate is not None
        return gated_nvfp4_qmv(x, gate, self.weight, self.scales, self.bank)
