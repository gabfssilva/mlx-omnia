"""The unrouted NVFP4 gate/up SwiGLU of one token, over a halved scale plane.

The single-weight sibling of `nvfp4_packed_gate_up`: no keys, no expert stride, and a
fused bank that is plain `[gate rows | up rows]` concatenation rather than the routed
bank's 32-row interleave, so a simdgroup's up row is its gate row plus `inner`. One
simdgroup owns one output row and both of its projections, which keeps the activation
block in registers for two dots instead of one and writes the SwiGLU product from lane 0.

The scale plane is the group-32 halved form of `shared.nvfp4` — one byte per 32 weights
behind the 128-byte patch header, addressed by `lane >> 1` — and the patch lane restores
the two spans the quantizer wrote twice: gate row 0 and up row 0 of the first k-block.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.mlp.kernel import Activation
from sideros.core.kernels.shared.nvfp4 import (
    QDOT_HEADER,
    SCALE_PATCH_BYTES,
    halved_group32_scales,
)
from sideros.core.layers import SwiGLU
from sideros.core.mxcompat import metal_kernel

_SOURCE = """
constexpr uint input_width = HIDDEN;
constexpr uint output_width = INNER;
constexpr uint packed_row_bytes = input_width / 2;
constexpr uint scale_row_bytes = input_width / 32;
constexpr uint scale_patch_bytes = PATCH;
constexpr uint block_width = 512;
constexpr uint values_per_lane = 16;

uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = tile * 2 + simd_group;

const device uint8_t* gate_row_weight =
    (const device uint8_t*)fused_weight +
    row * packed_row_bytes + lane * 8;
const device uint8_t* up_row_weight =
    (const device uint8_t*)fused_weight +
    (row + output_width) * packed_row_bytes + lane * 8;
const device uint8_t* gate_row_scale =
    fused_scales + scale_patch_bytes + row * scale_row_bytes + (lane >> 1);
const device uint8_t* up_row_scale =
    fused_scales + scale_patch_bytes + (row + output_width) * scale_row_bytes + (lane >> 1);

thread float gate_result = 0.0f;
thread float up_result = 0.0f;
thread float input_values[values_per_lane];

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

    gate_result += nvfp4_qdot_16(
        gate_row_weight + block / 2,
        input_values,
        nvfp4_scale((row == 0 && block == 0 && lane == 1)
            ? fused_scales[0] : gate_row_scale[block / 32]));
    up_result += nvfp4_qdot_16(
        up_row_weight + block / 2,
        input_values,
        nvfp4_scale((row == 0 && block == 0 && lane == 1)
            ? fused_scales[1] : up_row_scale[block / 32]));
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
    activated[row] = bfloat(silu * up);
}
"""

_KERNEL = metal_kernel(
    name="nvfp4_halved_gate_up_rows1",
    input_names=["input", "fused_weight", "fused_scales"],
    output_names=["activated"],
    source=_SOURCE,
    header=QDOT_HEADER,
)


def halve_gate_up_scales(fused_scales: mx.array) -> mx.array | None:
    """The fused `[2*inner, hidden/16]` plane halved in place — it is already in the
    kernel's read order, so only the patch header is new. Gate row 0 is flat pair 0 and up
    row 0 is flat pair `inner * hidden/32`, the two the kernel's patch lane restores from
    header slots 0 and 1. `None` when the checkpoint fails the certificate."""
    assert fused_scales.dtype == mx.uint8 and fused_scales.ndim == 2
    inner = fused_scales.shape[0] // 2
    pairs_per_row = fused_scales.shape[1] // 2
    return halved_group32_scales(fused_scales, (0, inner * pairs_per_row))


def nvfp4_halved_gate_up_applies(hidden: int, inner: int) -> bool:
    """The K loop steps a whole 512-value block with no tail, and a threadgroup writes the
    two rows its two simdgroups own."""
    return hidden % 512 == 0 and inner % 2 == 0


def nvfp4_halved_gate_up(
    x: mx.array, fused_weight: mx.array, fused_scales: mx.array
) -> mx.array:
    """x [hidden] bf16 through one gate‖up stack with SwiGLU applied: `fused_weight`
    [2*inner, hidden/8] uint32 as `[gate | up]`, `fused_scales` the 1-D plane from
    `halve_gate_up_scales` -> silu(gate)*up as [inner] bf16."""
    hidden = fused_weight.shape[1] * 8
    inner = fused_weight.shape[0] // 2
    assert nvfp4_halved_gate_up_applies(hidden, inner)
    assert x.dtype == mx.bfloat16 and x.size == hidden
    assert fused_weight.dtype == mx.uint32
    assert fused_scales.dtype == mx.uint8 and fused_scales.ndim == 1
    assert fused_scales.size == SCALE_PATCH_BYTES + 2 * inner * (hidden // 32)
    return _KERNEL(
        inputs=[x, fused_weight, fused_scales],
        template=[("HIDDEN", hidden), ("INNER", inner), ("PATCH", SCALE_PATCH_BYTES)],
        grid=((inner // 2) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(inner,)],
        output_dtypes=[mx.bfloat16],
    )[0]


@dataclass(frozen=True)
class Nvfp4Mlp:
    fused_weight: mx.array
    halved_scales: mx.array
    down: nn.Linear | nn.QuantizedLinear

    @classmethod
    def build(
        cls, leaf: SwiGLU, *, hidden: int, inner: int, activation: Activation
    ) -> Self | None:
        # Only the gate/up half has a kernel here; the down projection stays the
        # leaf's own call, so its format is free. The halving is a certificate on
        # the checkpoint's scale plane, paid once at construction.
        gate_up: nn.Module = leaf.gate_up_proj
        down: nn.Module = leaf.down_proj
        if activation != "silu":
            return None
        if not isinstance(gate_up, nn.QuantizedLinear) or gate_up.mode != "nvfp4":
            return None
        if not isinstance(down, nn.Linear | nn.QuantizedLinear):
            return None
        if not nvfp4_halved_gate_up_applies(hidden, inner):
            return None
        halved = halve_gate_up_scales(gate_up.scales)
        if halved is None:
            return None
        return cls(gate_up.weight, halved, down)

    def __call__(self, row: mx.array, residual: mx.array) -> mx.array:
        activated = nvfp4_halved_gate_up(row, self.fused_weight, self.halved_scales)
        return self.down(activated[None]).reshape(-1) + residual
