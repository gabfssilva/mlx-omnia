"""A dense bf16 MLP of a one-token step in two dispatches: fused gate‖up + SwiGLU, then
down + residual.

Transcribed from the mlxfast-challenge record tree (Layr Labs, MIT). Both halves are the
same mlx gemv shape -- one simdgroup owns four output rows, a lane owns four contiguous
columns, the contraction walks 128-wide blocks and a simd tree finishes each row -- with
the epilogue folded in so the projection's output never round-trips through DRAM.

What the fusion actually buys is the epilogue's dispatches, not the gemv: at T=1 both
kernels are at the bandwidth wall reading their weight, and the SwiGLU and the residual
add are three or four tiny elementwise ops each that cost a command per op.

The activation chain is bf16 throughout, deliberately: the accumulator is fp32, but gate
and up are rounded to bf16 before the sigmoid and every step of
`silu = gate / (1 + exp(-gate))` after it rounds again -- the numerically stable
`exp(|g|)` branch form, so a large positive gate cannot overflow the exponential. Anything
comparing against this has to round the same way at the same points; an fp32 SwiGLU is a
different function by a couple of bf16 ulps.

`fused_weight` is `concatenate([gate_proj.weight, up_proj.weight], axis=0)` -- gate output
rows first -- built once, not per step.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_GATE_UP_SOURCE = """
    constexpr uint in_vec_size = (uint)K;
    constexpr uint output_width = (uint)N;
    constexpr uint rows_per_thread = 4;
    constexpr uint values_per_thread = 4;
    constexpr uint block_width = 128;
    constexpr uint blocks = in_vec_size / block_width;
    constexpr uint rows_per_group = 64;

    uint tile = threadgroup_position_in_grid.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

    thread float gate_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float up_result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float coefficients[values_per_thread];

    uint column = lane * values_per_thread;
    for (uint block = 0; block < blocks; ++block) {
        const vec<bfloat, 4> c4 =
            *((const device vec<bfloat, 4>*)(input + column));
        for (uint i = 0; i < values_per_thread; ++i) {
            coefficients[i] = float(c4[i]);
        }
        for (uint row = 0; row < rows_per_thread; ++row) {
            const device vec<bfloat, 4>* gate_row_values =
                (const device vec<bfloat, 4>*)(
                    fused_weight + (row_base + row) * in_vec_size + column);
            const vec<bfloat, 4> gw = gate_row_values[0];
            const device vec<bfloat, 4>* up_row_values =
                (const device vec<bfloat, 4>*)(
                    fused_weight +
                    (output_width + row_base + row) * in_vec_size + column);
            const vec<bfloat, 4> uw = up_row_values[0];
            for (uint i = 0; i < values_per_thread; ++i) {
                gate_result[row] += float(gw[i]) * coefficients[i];
                up_result[row] += float(uw[i]) * coefficients[i];
            }
        }
        column += block_width;
    }

    for (uint row = 0; row < rows_per_thread; ++row) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            gate_result[row] +=
                metal::simd_shuffle_down(gate_result[row], delta);
            up_result[row] +=
                metal::simd_shuffle_down(up_result[row], delta);
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < rows_per_thread; ++row) {
            bfloat gate = bfloat(gate_result[row]);
            bfloat up = bfloat(up_result[row]);
            bfloat exp_abs = metal::exp(metal::abs(gate));
            bfloat denominator = bfloat(1) + exp_abs;
            bfloat y = bfloat(1) / denominator;
            bfloat sigmoid = gate < bfloat(0) ? y : bfloat(1) - y;
            bfloat silu = bfloat(gate * sigmoid);
            activated[row_base + row] = bfloat(silu * up);
        }
    }
"""

_DOWN_SOURCE = """
    constexpr uint in_vec_size = (uint)K;
    constexpr uint rows_per_thread = 4;
    constexpr uint values_per_thread = 4;
    constexpr uint block_width = 128;
    constexpr uint blocks = in_vec_size / block_width;
    constexpr uint rows_per_group = 16;

    uint tile = threadgroup_position_in_grid.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;

    uint row_base = tile * rows_per_group + simd_group * rows_per_thread;

    thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float coefficients[values_per_thread];

    uint column = lane * values_per_thread;
    for (uint block = 0; block < blocks; ++block) {
        const vec<bfloat, 4> c4 =
            *((const device vec<bfloat, 4>*)(activated + column));
        for (uint i = 0; i < values_per_thread; ++i) {
            coefficients[i] = float(c4[i]);
        }
        for (uint row = 0; row < rows_per_thread; ++row) {
            const device vec<bfloat, 4>* row_values =
                (const device vec<bfloat, 4>*)(
                    down_weight + (row_base + row) * in_vec_size + column);
            const vec<bfloat, 4> w = row_values[0];
            for (uint i = 0; i < values_per_thread; ++i) {
                result[row] += float(w[i]) * coefficients[i];
            }
        }
        column += block_width;
    }

    for (uint row = 0; row < rows_per_thread; ++row) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            result[row] += metal::simd_shuffle_down(result[row], delta);
        }
    }
    if (lane == 0) {
        for (uint row = 0; row < rows_per_thread; ++row) {
            bfloat down = bfloat(result[row]);
            output[row_base + row] =
                bfloat(residual[row_base + row] + down);
        }
    }
"""

_GATE_UP_KERNEL = metal_kernel(
    name="dense_gate_up_swiglu_bf16",
    input_names=["input", "fused_weight"],
    output_names=["activated"],
    source=_GATE_UP_SOURCE,
)

_DOWN_KERNEL = metal_kernel(
    name="dense_down_residual_bf16",
    input_names=["activated", "down_weight", "residual"],
    output_names=["output"],
    source=_DOWN_SOURCE,
)

_GATE_UP_ROWS_PER_TG = 64
_GATE_UP_THREADS = 512
_DOWN_ROWS_PER_TG = 16
_DOWN_THREADS = 128


def dense_gate_up_swiglu_applies(hidden: int, inner: int, dtype: mx.Dtype) -> bool:
    """A lane loads `vec<bfloat,4>` from both the activation and two weight rows, a
    simdgroup steps the contraction 128 columns at a time and a threadgroup writes 64
    output rows, none of it bounds-checked: the contraction has to be a whole number of
    128-wide blocks and the output width a whole number of threadgroups. bf16 only -- the
    loads, the SwiGLU and the store all name the type."""
    return dtype == mx.bfloat16 and hidden % 128 == 0 and inner % _GATE_UP_ROWS_PER_TG == 0


def dense_down_residual_applies(hidden: int, inner: int, dtype: mx.Dtype) -> bool:
    """Same tiling with a 16-row threadgroup: the `inner` contraction covers whole 128-wide
    blocks and the `hidden` output axis whole threadgroups."""
    return dtype == mx.bfloat16 and inner % 128 == 0 and hidden % _DOWN_ROWS_PER_TG == 0


def dense_gate_up_swiglu(x: mx.array, fused_weight: mx.array) -> mx.array:
    """x [hidden] through the stacked gate‖up weight with SwiGLU applied: `fused_weight`
    [2·inner, hidden] with the gate rows first -> [inner]."""
    inner = fused_weight.shape[0] // 2
    hidden = fused_weight.shape[1]
    assert x.size == hidden
    assert dense_gate_up_swiglu_applies(hidden, inner, x.dtype)
    return _GATE_UP_KERNEL(
        inputs=[x, fused_weight],
        template=[("K", hidden), ("N", inner)],
        grid=(inner // _GATE_UP_ROWS_PER_TG * _GATE_UP_THREADS, 1, 1),
        threadgroup=(_GATE_UP_THREADS, 1, 1),
        output_shapes=[(inner,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def dense_down_residual(
    activated: mx.array, down_weight: mx.array, residual: mx.array
) -> mx.array:
    """activated [inner] down-projected and residual-added in one dispatch: `down_weight`
    [hidden, inner] -> [hidden]. The add rounds to bf16 once, where the op chain it
    replaces rounds."""
    hidden, inner = down_weight.shape
    assert activated.size == inner and residual.size == hidden
    assert dense_down_residual_applies(hidden, inner, activated.dtype)
    return _DOWN_KERNEL(
        inputs=[activated, down_weight, residual],
        template=[("K", inner)],
        grid=(hidden // _DOWN_ROWS_PER_TG * _DOWN_THREADS, 1, 1),
        threadgroup=(_DOWN_THREADS, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[mx.bfloat16],
    )[0]
