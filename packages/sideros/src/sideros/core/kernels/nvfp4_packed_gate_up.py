"""The routed gate/up SwiGLU of one token, with routing decided inside the kernel.

One simdgroup owns one output row of one selected expert, and the threadgroup grid is
`top_k x inner/2`: the expert a threadgroup works on is not read from an index buffer, it
is *recomputed* from the router's ordinal keys. Each lane holds `experts / 32` of them
(expert `lane + 32j`), and one round of `router_topk_extract_round` is an argmin over that
register file followed by a five-step butterfly — the (ordinal, index) pair travels as a
`uint2` so both components come from the same source lane — leaving every lane holding the
same winner, which the owning lane then masks out of its register. Threadgroup `s` runs
`s + 1` rounds and keeps the last. That is `O(k^2 / 32)` register work per threadgroup,
paid to delete the argpartition dispatch and the gather it feeds: no index buffer is ever
written, and no dependent read stalls the weight loads, which start immediately.

The tie rule is the sort's, not a reduction's: equal ordinals resolve to the lower expert
index, in `router_ordinal_before`, at every comparison and at every butterfly step.

Two orderings meet in the weight addressing. The fused bank interleaves gate and up in
tiles of 32 rows (`gate_row = (row / 32) * 64 + row % 32`, up at `+32`), which the kernel
walks directly; the scale plane is a side copy already permuted into the kernel's walk
order — per expert `[tile of 4 rows][k-block][sub][16 bytes]`, `sub` being the four rows
times gate/up — so a lane's scale byte is one address computed from `logical_row / 4` and
`logical_row % 4`, and adjacent k-blocks are 128 bytes apart instead of a row apart. The
codes stay where they are; only the ~16 MiB of scales are copied at load time.

The K loop is software-pipelined by hand: the next k-block's codes and scale bytes are
issued before the current block's dot products are computed, so the two loads of the next
iteration are already in flight while the ALU works. `nvfp4_packed` covers the decode and
the `4194304.0f` at the bf16 boundary.
"""

import mlx.core as mx

from sideros.core.kernels.nvfp4_packed import (
    QDOT_HEADER,
    SCALE_PATCH_BYTES,
    halved_group32_scales,
)
from sideros.core.kernels.router_ordinal import ORDINAL_HEADER
from sideros.core.mxcompat import metal_kernel

_TOURNAMENT_HEADER = """
template <uint keys_per_lane, uint expert_count>
METAL_FUNC uint router_topk_extract_round(
    thread const uint* keys, thread uint& mask, uint lane) {
    uint best_ordinal = 0xFFFFFFFFu;
    uint best_index = expert_count;
    for (uint j = 0; j < keys_per_lane; ++j) {
        if ((mask & (1u << j)) != 0u) continue;
        uint e = lane + 32u * j;
        uint o = keys[j];
        if (router_ordinal_before(o, e, best_ordinal, best_index)) {
            best_ordinal = o;
            best_index = e;
        }
    }
    // Transport the comparator's (ordinal, expert-index) state as one uint2
    // through each butterfly step. simd_shuffle_xor moves both components
    // bit-for-bit from the same source lane; comparator order is unchanged.
    uint2 best_pair = uint2(best_ordinal, best_index);
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const uint2 other_pair = simd_shuffle_xor(best_pair, offset);
        if (router_ordinal_before(
            other_pair.x, other_pair.y, best_pair.x, best_pair.y)) {
            best_pair = other_pair;
        }
    }
    best_index = best_pair.y;
    if ((best_index & 31u) == lane) {
        mask |= 1u << (best_index >> 5u);
    }
    return best_index;
}
"""

_SOURCE = """
constexpr uint input_width = HIDDEN;
constexpr uint output_width = INNER;
constexpr uint block_width = 512;
constexpr uint values_per_lane = 16;
constexpr uint routed_experts = TOPK;
constexpr uint expert_count = EXPERTS;
constexpr uint keys_per_lane = expert_count / 32;
constexpr uint fused_row_bytes = input_width / 2;
constexpr uint fused_expert_bytes = 2 * output_width * fused_row_bytes;
constexpr uint scale_patch_bytes = PATCH;
constexpr uint scale_row_bytes = 16;
constexpr uint scale_sub_bytes = 8 * scale_row_bytes;
constexpr uint scale_kblock_bytes = scale_sub_bytes;
constexpr uint scale_tile_bytes = (input_width / block_width) * scale_kblock_bytes;
constexpr uint packed_expert_bytes = (output_width / 4) * scale_tile_bytes;

uint group = threadgroup_position_in_grid.x;
uint expert_slot = group % routed_experts;
uint tile = group / routed_experts;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint logical_row = tile * 2 + simd_group;
thread uint top8_keys[keys_per_lane];
    for (uint j = 0; j < keys_per_lane; ++j) {
        top8_keys[j] = router_keys[lane + 32u * j];
    }
    uint top8_mask = 0u;
    uint top8_winner = 0u;
    for (uint r = 0; r <= expert_slot; ++r) {
        top8_winner = router_topk_extract_round<keys_per_lane, expert_count>(
            top8_keys, top8_mask, lane);
    }
uint expert = top8_winner;

const device uint8_t* expert_weight =
    (const device uint8_t*)fused_weight + expert * fused_expert_bytes;
const device uint8_t* row_scales =
    packed_scales + scale_patch_bytes + expert * packed_expert_bytes
    + (logical_row / 4) * scale_tile_bytes;
uint sub = logical_row % 4;
uint gate_row = (logical_row / 32) * 64 + logical_row % 32;
uint up_row = gate_row + 32;

thread float gate_result = 0.0f;
thread float up_result = 0.0f;
thread float input_values[values_per_lane];

uint2 gate_codes;
uint2 up_codes;
uint8_t gate_sb;
uint8_t up_sb;
{
    const device uint8_t* first_scales =
        row_scales + sub * 2 * scale_row_bytes + (lane >> 1);
    bool patch_lane = expert == 0 && logical_row == 0 && lane == 1;
    gate_sb = patch_lane ? packed_scales[0] : first_scales[0];
    up_sb = patch_lane ? packed_scales[1] : first_scales[scale_row_bytes];
    gate_codes = *(const device uint2*)(
        expert_weight + gate_row * fused_row_bytes + lane * 8);
    up_codes = *(const device uint2*)(
        expert_weight + up_row * fused_row_bytes + lane * 8);
}

for (uint block = 0; block < input_width; block += block_width) {
    const device vec<bfloat, 4>* input_vectors =
        (const device vec<bfloat, 4>*) (
            input + block + lane * values_per_lane);
    for (uint i = 0; i < values_per_lane / 4; ++i) {
        const vec<bfloat, 4> values = input_vectors[i];
        input_values[4 * i] = values[0];
        input_values[4 * i + 1] = values[1];
        input_values[4 * i + 2] = values[2];
        input_values[4 * i + 3] = values[3];
    }

    const uint2 cur_gate_codes = gate_codes;
    const uint2 cur_up_codes = up_codes;
    const uint8_t cur_gate_sb = gate_sb;
    const uint8_t cur_up_sb = up_sb;
    const uint next_block = block + block_width;
    if (next_block < input_width) {
        const device uint8_t* next_scales =
            row_scales + (next_block / block_width) * scale_kblock_bytes
            + sub * 2 * scale_row_bytes + (lane >> 1);
        gate_sb = next_scales[0];
        up_sb = next_scales[scale_row_bytes];
        gate_codes = *(const device uint2*)(
            expert_weight + gate_row * fused_row_bytes
            + next_block / 2 + lane * 8);
        up_codes = *(const device uint2*)(
            expert_weight + up_row * fused_row_bytes
            + next_block / 2 + lane * 8);
    }

    gate_result += nvfp4_qdot_codes_16(
        cur_gate_codes, input_values,
        nvfp4_scale(cur_gate_sb));
    up_result += nvfp4_qdot_codes_16(
        cur_up_codes, input_values,
        nvfp4_scale(cur_up_sb));
}

gate_result = simd_sum(gate_result);
up_result = simd_sum(up_result);
if (lane == 0) {
    bfloat gate = bfloat(gate_result * 4194304.0f);
    bfloat up = bfloat(up_result * 4194304.0f);
    bfloat exp_abs = metal::exp(metal::abs(gate));
    bfloat denominator = bfloat(1) + exp_abs;
    bfloat y = bfloat(1) / denominator;
    bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
    bfloat silu = bfloat(gate * sigmoid);
    activated[expert_slot * output_width + logical_row] =
        bfloat(silu * up);
}
"""

_KERNEL = metal_kernel(
    name="nvfp4_packed_gate_up_keys",
    input_names=["input", "fused_weight", "packed_scales", "router_keys"],
    output_names=["activated"],
    source=_SOURCE,
    header=QDOT_HEADER + "\n" + ORDINAL_HEADER + "\n" + _TOURNAMENT_HEADER,
)


def pack_gate_up_scales(fused_scales: mx.array) -> mx.array | None:
    """The fused gate/up scale plane `[experts, 2*inner, hidden/16]` as the bank the
    kernel addresses: bytes are reordered, never recomputed.

    Per expert the walk order is `[tile of 4 logical rows][k-block][sub][16 bytes]` with
    `sub = (logical_row % 4) * 2 + {0 gate, 1 up}`, which bakes the fused bank's 32-row
    gate/up interleave into scale storage order. The halving then collapses each 32-byte
    row-block to the 16 even bytes; the walk order puts expert 0's gate row 0 at row-block
    0 and its up row 0 at row-block 1, so the two spans the quantizer may leave unequal are
    flat pairs 0 and 16 and land in header slots 0 and 1 — where the kernel's patch lane
    reads them. `None` when the checkpoint fails that certificate."""
    assert fused_scales.dtype == mx.uint8 and fused_scales.ndim == 3
    experts = fused_scales.shape[0]
    rows = fused_scales.shape[1]
    kblocks = fused_scales.shape[2] // 32
    assert rows % 8 == 0 and fused_scales.shape[2] == kblocks * 32

    order: list[int] = []
    for tile in range(rows // 8):
        for kblock in range(kblocks):
            for sub in range(8):
                logical_row = tile * 4 + sub // 2
                gate_row = (logical_row // 32) * 64 + logical_row % 32
                fused_row = gate_row if sub % 2 == 0 else gate_row + 32
                order.append(fused_row * kblocks + kblock)

    row_blocks = fused_scales.reshape(experts, rows * kblocks, 32)
    packed = mx.take(row_blocks, mx.array(order, dtype=mx.int32), axis=1)
    return halved_group32_scales(packed, (0, 16))


def nvfp4_packed_gate_up_applies(hidden: int, inner: int, experts: int, top_k: int) -> bool:
    """A lane covers 16 values and the K loop steps a whole 512-value block, so the
    contraction is blocked, not guarded: `hidden` must be a multiple of 512. The fused
    bank's gate/up interleave is 32 rows wide and the scale tile is four rows, so `inner`
    must be a multiple of 32. The tournament gives each lane `experts / 32` keys and one
    mask bit each, so `experts` is a multiple of 32 and at most 1024, and a threadgroup
    exists per selected slot."""
    return (
        hidden % 512 == 0
        and inner % 32 == 0
        and experts % 32 == 0
        and 0 < experts // 32 <= 32
        and 0 < top_k <= experts
    )


def nvfp4_packed_gate_up(
    x: mx.array,
    fused_weight: mx.array,
    packed_scales: mx.array,
    router_keys: mx.array,
    top_k: int,
) -> mx.array:
    """x [hidden] bf16 through the `top_k` experts the ordinal keys select, with SwiGLU
    applied: `fused_weight` [experts, 2*inner, hidden/8] uint32 in 32-row gate/up tiles,
    `packed_scales` the 1-D bank from `pack_gate_up_scales`, `router_keys` [experts] uint32
    -> silu(gate)*up as [top_k, inner] bf16, slot `s` holding the `s`-th best expert."""
    hidden = fused_weight.shape[2] * 8
    inner = fused_weight.shape[1] // 2
    experts = router_keys.shape[0]
    assert nvfp4_packed_gate_up_applies(hidden, inner, experts, top_k)
    assert x.dtype == mx.bfloat16 and x.size == hidden
    assert fused_weight.dtype == mx.uint32 and fused_weight.shape[0] == experts
    assert packed_scales.dtype == mx.uint8 and packed_scales.ndim == 1
    assert packed_scales.size == SCALE_PATCH_BYTES + experts * 2 * inner * (hidden // 32)
    assert router_keys.dtype == mx.uint32
    return _KERNEL(
        inputs=[x, fused_weight, packed_scales, router_keys],
        template=[
            ("HIDDEN", hidden),
            ("INNER", inner),
            ("TOPK", top_k),
            ("EXPERTS", experts),
            ("PATCH", SCALE_PATCH_BYTES),
        ],
        grid=(top_k * (inner // 2) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(top_k, inner)],
        output_dtypes=[mx.bfloat16],
    )[0]
