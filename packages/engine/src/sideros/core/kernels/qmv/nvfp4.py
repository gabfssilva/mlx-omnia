"""One-row NVFP4 projection reading a lane-major group-scale bank, in one dispatch.

Transcribed from the mlxfast-challenge (Layr Labs, MIT) record tree's
`laguna_decode_nvfp4_qkv_*_r1_v1_lm1_pw1_se1_sd1` -- the lane-major, pairwise,
seed-elided, scale-deferred arm of its decode-time NVFP4 projection. Every line is the
record's, down to its indentation; the only edits are the contraction width becoming a
template argument and the two header helpers losing their model-named prefix. It means
`mx.quantized_matmul(..., transpose=True, group_size=16, bits=4, mode="nvfp4")` at T=1,
with four decode-path transforms the record tree certifies as bit-exact:

- **Split-nibble decode.** Eight e2m1 codes become four `half2` bit patterns through
  three mask constants instead of eight shift+mask pairs. Every form is an OR of masked
  shifts, so the eight halves are the same bits mlx's `fp_qmv_fast` builds.
- **Scale fold and scale defer.** The e4m3 byte shifted into a half is the true scale
  divided by 256, and the `0x8E00`-masked e2m1 halves are the true codes times `2^14`;
  both powers of two are constant across the row, so the `2^22` they compose to is
  hoisted out of the K loop and applied once, after `simd_sum`.
- **Sign-domain specialization.** `bits & 128` is zero on every scale byte a census of
  the record's checkpoint could reach, so the sign select degenerates to the identity
  and the decode is `ushort(bits) << 7`. A negative scale byte would decode as its
  magnitude here -- `nvfp4_qmv_applies` cannot see the plane, so the caller owns that.
- **Seed elision.** The first code word assigns the dot accumulator instead of being
  added to a dead `+0.0f`. Only an all-`-0.0` group's zero can flip sign, which the
  `+0.0f`-seeded `result` and the bf16 epilogue absorb.

The scale bank is `shared.nvfp4`'s lane-major packing: one base byte and a 4-bit
offset per group, in lane order, so a lane's four blocks arrive as one `ushort`. That
single 2-byte load is what pins the contraction to four 512-value blocks -- four
nibbles is all a `ushort` holds -- and `nvfp4_qmv_applies` says so. Rows the bank could
not fit carry base `0xFF` and take the else arm, which reads the stock plane at the
stock stride.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.qmv.kernel import Epilogue, QmvLeaf
from sideros.core.kernels.shared.nvfp4 import LaneMajorScales, lane_major_scales
from sideros.core.mxcompat import metal_kernel

_BLOCK_SIZE = 512
_BLOCKS_PER_USHORT = 4
_GROUP_SIZE = 16
_BITS = 4

_HEADER = """
static inline float nvfp4_tail_scale(uint8_t bits) {
    ushort raw = ushort(bits) << 7;
        return float(as_type<half>(raw));
}

static inline float nvfp4_tail_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale
) {
    float accum;
    const device uint2* wq = (const device uint2*)w;
    const uint2 codes = wq[0];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const uint32_t c = (j == 0) ? codes.x : codes.y;
        // Split-nibble decode: the same eight `half` bit patterns per
        // code word as the original shift+mask sequence, in fewer
        // integer ops with three mask constants instead of eight — the
        // form the current stock `fp_qmv_fast` compiles (every form is
        // an OR of masked shifts, so the decode is bit-identical).
        const uint32_t xe = c & 0x0F0F0F0Fu;
        const uint32_t ge = xe | (xe << 3);
        const uint32_t yo = c & 0xF0F0F0F0u;
        const uint32_t go = yo | (yo >> 3);
        const uint32_t p0 = (ge << 9) & 0x8E008E00u;
        const uint32_t p1 = (go << 8) & 0x8E008E00u;
        const uint32_t p2 = (ge << 1) & 0x8E008E00u;
        const uint32_t p3 = go & 0x8E008E00u;
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
    return scale * accum;
}
"""

_SOURCE = """
constexpr uint axis_size = (uint)AXIS;
constexpr uint num_simdgroups = 2;
constexpr uint values_per_thread = 16;
constexpr uint block_size = 512;
constexpr uint in_vec_size_w = axis_size / 2;
constexpr uint in_vec_size_g = axis_size / 16;
constexpr uint blocks_per_row = in_vec_size_g / 32;

uint tile = threadgroup_position_in_grid.x;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
uint out_row = tile * num_simdgroups + simd_gid;

const device uint8_t* ws = (const device uint8_t*)weight_codes +
    out_row * in_vec_size_w + simd_lid * 8;

thread uint8_t sb[blocks_per_row];
const uint8_t row_base = scale_bases[out_row];
if (row_base != 0xFFu) {
    const device ushort* nb = (const device ushort*)(
        scale_nibbles + out_row * (in_vec_size_g / 4))
        + (simd_lid >> 1);
    const ushort packed = nb[0];
#pragma unroll
    for (uint b = 0; b < blocks_per_row; ++b) {
        sb[b] = uint8_t(row_base + ((packed >> (b << 2)) & 0x0Fu));
    }
} else {
    const device uint8_t* sc = weight_scales +
        out_row * in_vec_size_g + simd_lid;
#pragma unroll
    for (uint b = 0; b < blocks_per_row; ++b) {
        sb[b] = sc[b * (block_size / 16)];
    }
}

thread float x_thread[values_per_thread];
thread float result = 0.0f;

uint column = simd_lid * values_per_thread;
for (uint k = 0; k < axis_size; k += block_size) {
    for (uint i = 0; i < values_per_thread; ++i) {
        x_thread[i] = float(normalized[column + i]);
    }
    result += nvfp4_tail_qdot(
        ws, x_thread, nvfp4_tail_scale(sb[k / block_size]));
    ws += block_size / 2;
    column += block_size;
}

result = simd_sum(result * 4194304.0f);
if (simd_lid == 0) {
    projected[out_row] = bfloat(result);
}
"""

_KERNEL = metal_kernel(
    name="nvfp4_qmv_lane_major",
    input_names=[
        "normalized",
        "weight_codes",
        "scale_nibbles",
        "scale_bases",
        "weight_scales",
    ],
    output_names=["projected"],
    source=_SOURCE,
    header=_HEADER,
)


def nvfp4_qmv_applies(kdim: int, rows: int, *, group_size: int, bits: int) -> bool:
    """The contraction has to tile exactly and the lane's scale run has to fit one load.

    `bits`/`group_size` are pinned to 4/16 -- one `uint2` per lane is one whole NVFP4
    group -- and a simdgroup steps 512 values at a time with nothing bounds-checked. The
    binding constraint is the lane-major bank: a lane's whole row of 4-bit scale offsets
    is read as a single `ushort`, which holds exactly four, so the contraction is exactly
    four blocks wide. Two simdgroups per threadgroup take one output row each.
    """
    return (
        bits == _BITS
        and group_size == _GROUP_SIZE
        and kdim == _BLOCK_SIZE * _BLOCKS_PER_USHORT
        and rows % 2 == 0
    )


def nvfp4_qmv(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    bank: LaneMajorScales,
) -> mx.array:
    """x [..., kdim] (one row, bf16) against a [rows, kdim] NVFP4 stack -> [..., rows].

    `scales` is the stock `[rows, kdim / 16]` e4m3 plane and `bank` its lane-major
    packing; both stay bound because an escaped row reads the plane.
    """
    kdim = weight.shape[-1] * 8
    rows = weight.shape[-2]
    assert x.dtype == mx.bfloat16 and x.size == kdim
    assert nvfp4_qmv_applies(kdim, rows, group_size=_GROUP_SIZE, bits=_BITS)
    assert scales.shape == (rows, kdim // _GROUP_SIZE)
    assert bank.nibbles.shape == (rows, kdim // (4 * _GROUP_SIZE))
    assert bank.bases.shape == (rows,)
    return _KERNEL(
        inputs=[x, weight, bank.nibbles, bank.bases, scales],
        template=[("AXIS", kdim)],
        grid=((rows // 2) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.bfloat16],
    )[0]


@dataclass(frozen=True)
class Nvfp4Qmv:
    weight: mx.array
    scales: mx.array
    bank: LaneMajorScales

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
        if epilogue != "none" or not isinstance(leaf, nn.QuantizedLinear) or "bias" in leaf:
            return None
        if (leaf.mode, leaf.group_size, leaf.bits) != ("nvfp4", _GROUP_SIZE, _BITS):
            return None
        if not nvfp4_qmv_applies(kdim, rows, group_size=leaf.group_size, bits=leaf.bits):
            return None
        bank = lane_major_scales(leaf.scales)
        mx.eval(bank.nibbles, bank.bases)
        return cls(leaf.weight, leaf.scales, bank)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        return nvfp4_qmv(x, self.weight, self.scales, self.bank)
