"""Every down projection of a MoE block, its routing sum and the residual add, in one
dispatch.

The threadgroup is `top_k + 1` simdgroups wide and the last one is not routed: slot `s`
does expert `indices[s]` over `routed_activated[s]`, slot `top_k` does the unrouted stack
over its own activations, and the only difference between them is which base pointer and
which scale plane the same four row dots read. Four output rows per simdgroup means the
activation block is loaded once per threadgroup-column and reused four times, and the
whole reduction over experts happens in `(top_k + 1) * 4` bf16 words of threadgroup memory
behind a single barrier — no partial-output buffer, no second kernel, no atomics.

There is no K loop: a lane's 16 values times 32 lanes is the entire contraction, which is
what pins `inner` at 512. The rows are staged before they are consumed — all four
`uint2` code words and all four scale bytes are issued, then the four dots run — so four
independent loads are in flight instead of one load feeding one dot.

The epilogue is bf16 the whole way and its order is load-bearing: products accumulate
into `routed_total` slot by slot in ascending order, each rounding once, and the routed
scale multiplies the total, not each term. `nvfp4_packed` covers the decode and the
`4194304.0f`; the routed scaling factor is the model's, so it arrives as an input.
"""

import mlx.core as mx

from sideros.core.kernels.nvfp4_packed import (
    QDOT_HEADER,
    SCALE_PATCH_BYTES,
    halved_group32_scales,
)
from sideros.core.mxcompat import metal_kernel

_SOURCE = """
constexpr uint input_width = INNER;
constexpr uint output_width = HIDDEN;
constexpr uint routed_experts = TOPK;
constexpr uint shared_slot = routed_experts;
constexpr uint outputs_per_simd = 4;
constexpr uint values_per_lane = 16;
constexpr uint packed_row_bytes = input_width / 2;
constexpr uint scale_patch_bytes = PATCH;
constexpr uint shared_scale_row_bytes = input_width / 32;
constexpr uint routed_scale_row_bytes = input_width / 32;
constexpr uint packed_expert_bytes =
    output_width * packed_row_bytes;
constexpr uint scale_expert_bytes =
    output_width * routed_scale_row_bytes;

uint tile = threadgroup_position_in_grid.x;
uint slot = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint first_row = tile * outputs_per_simd;
bool is_shared = slot == shared_slot;
uint expert = is_shared ? 0 : uint(indices[slot]);

const device bfloat* expert_input = is_shared
    ? shared_activated
    : routed_activated + slot * input_width;
const device uint8_t* expert_weight = is_shared
    ? (const device uint8_t*)shared_down_weight
    : (const device uint8_t*)routed_down_weight +
        expert * packed_expert_bytes;
const device uint8_t* expert_scales = is_shared
    ? shared_down_scales + scale_patch_bytes
    : routed_down_scales + scale_patch_bytes
        + expert * scale_expert_bytes;
uint scale_row_bytes =
    is_shared ? shared_scale_row_bytes : routed_scale_row_bytes;
uint scale_lane = (lane >> 1);

thread float input_values[values_per_lane];
const device vec<bfloat, 4>* input_vectors =
    (const device vec<bfloat, 4>*)(
        expert_input + lane * values_per_lane);
for (uint i = 0; i < values_per_lane / 4; ++i) {
    const vec<bfloat, 4> values = input_vectors[i];
    input_values[4 * i] = values[0];
    input_values[4 * i + 1] = values[1];
    input_values[4 * i + 2] = values[2];
    input_values[4 * i + 3] = values[3];
}

thread float result[outputs_per_simd] = {0.0f};
uint2 row_codes[outputs_per_simd];
uint8_t row_sb[outputs_per_simd];
for (uint row = 0; row < outputs_per_simd; ++row) {
    uint output_row = first_row + row;
    row_codes[row] = *(const device uint2*)(
        expert_weight + output_row * packed_row_bytes + lane * 8);
    const device uint8_t* scale =
        expert_scales + output_row * scale_row_bytes + scale_lane;
    row_sb[row] =
        (output_row == 0 && lane == 1 && (is_shared || expert == 0))
        ? (is_shared ? shared_down_scales[0] : routed_down_scales[0])
        : scale[0];
}
for (uint row = 0; row < outputs_per_simd; ++row) {
    result[row] = nvfp4_qdot_codes_16(
        row_codes[row],
        input_values,
        nvfp4_scale(row_sb[row]));
    result[row] = simd_sum(result[row]);
}

threadgroup bfloat down_outputs[
    (routed_experts + 1) * outputs_per_simd
];
if (lane == 0) {
    for (uint row = 0; row < outputs_per_simd; ++row) {
        down_outputs[slot * outputs_per_simd + row] =
            bfloat(result[row] * 4194304.0f);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (slot == 0 && lane < outputs_per_simd) {
    bfloat routed_total = bfloat(0);
    for (uint routed_slot = 0;
         routed_slot < routed_experts;
         ++routed_slot) {
        bfloat route_weight =
            bfloat(router_weights[routed_slot]);
        bfloat product = bfloat(
            down_outputs[
                routed_slot * outputs_per_simd + lane
            ] * route_weight);
        routed_total = bfloat(product + routed_total);
    }
    bfloat routed = bfloat(
        routed_total * bfloat(routed_scaling[0]));
    bfloat shared =
        down_outputs[shared_slot * outputs_per_simd + lane];
    bfloat r2 = bfloat(routed + shared);
    output[first_row + lane] =
        bfloat(residual[first_row + lane] + r2);
}
"""

_KERNEL = metal_kernel(
    name="nvfp4_fused_down_residual",
    input_names=[
        "routed_activated",
        "routed_down_weight",
        "routed_down_scales",
        "indices",
        "router_weights",
        "shared_activated",
        "shared_down_weight",
        "shared_down_scales",
        "residual",
        "routed_scaling",
    ],
    output_names=["output"],
    source=_SOURCE,
    header=QDOT_HEADER,
)


def halve_down_scales(scales: mx.array) -> mx.array | None:
    """A down-projection scale plane halved in place: it is row-major in the kernel's read
    order already, so only the header is new, and the one span the quantizer wrote twice is
    flat pair 0 — row 0 of the first expert. `None` when the checkpoint fails the
    certificate."""
    assert scales.dtype == mx.uint8
    return halved_group32_scales(scales, (0,))


def nvfp4_fused_down_residual_applies(hidden: int, inner: int, top_k: int) -> bool:
    """A lane's 16 values times 32 lanes is the whole contraction — there is no K loop, so
    `inner` is exactly 512 and the routed and unrouted stacks must share it. A threadgroup
    writes four output rows and runs one simdgroup per routed slot plus one for the
    unrouted stack, which the 1024-thread limit caps at 31 routed slots."""
    return hidden % 4 == 0 and inner == 512 and 0 < top_k <= 31


def nvfp4_fused_down_residual(
    routed_activated: mx.array,
    routed_down_weight: mx.array,
    routed_down_scales: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    shared_activated: mx.array,
    shared_down_weight: mx.array,
    shared_down_scales: mx.array,
    residual: mx.array,
    routed_scaling: float,
) -> mx.array:
    """`routed_activated` [top_k, inner] bf16 down-projected through `indices` [top_k]
    uint32, weighted by `router_weights` [top_k] fp32, scaled by `routed_scaling` and added
    to the unrouted stack's own down projection and to `residual` [hidden] bf16 -> [hidden]
    bf16. Both weights are [.., hidden, inner/8] uint32 with their scale planes in the 1-D
    halved form of `halve_down_scales`."""
    hidden = routed_down_weight.shape[1]
    inner = routed_down_weight.shape[2] * 8
    top_k = indices.shape[0]
    assert nvfp4_fused_down_residual_applies(hidden, inner, top_k)
    assert routed_activated.dtype == mx.bfloat16 and routed_activated.size == top_k * inner
    assert routed_down_weight.dtype == mx.uint32
    experts = routed_down_weight.shape[0]
    assert routed_down_scales.dtype == mx.uint8 and routed_down_scales.ndim == 1
    assert routed_down_scales.size == SCALE_PATCH_BYTES + experts * hidden * (inner // 32)
    assert indices.dtype == mx.uint32
    assert router_weights.dtype == mx.float32 and router_weights.size == top_k
    assert shared_activated.dtype == mx.bfloat16 and shared_activated.size == inner
    assert shared_down_weight.dtype == mx.uint32
    assert shared_down_weight.shape == (hidden, inner // 8)
    assert shared_down_scales.dtype == mx.uint8 and shared_down_scales.ndim == 1
    assert shared_down_scales.size == SCALE_PATCH_BYTES + hidden * (inner // 32)
    assert residual.dtype == mx.bfloat16 and residual.size == hidden
    threads = (top_k + 1) * 32
    return _KERNEL(
        inputs=[
            routed_activated,
            routed_down_weight,
            routed_down_scales,
            indices,
            router_weights,
            shared_activated,
            shared_down_weight,
            shared_down_scales,
            residual,
            mx.array([routed_scaling], dtype=mx.float32),
        ],
        template=[
            ("HIDDEN", hidden),
            ("INNER", inner),
            ("TOPK", top_k),
            ("PATCH", SCALE_PATCH_BYTES),
        ],
        grid=(hidden // 4 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[mx.bfloat16],
    )[0]
