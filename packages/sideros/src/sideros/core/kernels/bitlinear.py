"""Ternary matmul over uint8-packed 1.58-bit weights, ported from mlx-lm's
``make_bitlinear_kernel`` (git main, ``bitlinear_layers.py``).

The weight is packed 4 ternary values per byte along the output axis:
``packed[p, c]`` holds the four outputs ``p, p + out/4, p + 2*out/4, p + 3*out/4``
at input column ``c`` (bits 0-1, 2-3, 4-5, 6-7, LSB-first, field ``- 1`` -> ``{-1,0,1}``).
One threadgroup owns one (output-pack, batch-row); its 32 lanes split the
``in_features`` reduction (``BLOCK * M = 128`` columns per super-step) and close it
with ``simd_sum``; lane 0 writes the four scaled outputs.

The kernel is a pure float matmul over the unpacked ternary weights followed by a
single scalar ``weight_scale``. The per-token int8 activation fake-quant that
transformers' ``AutoBitLinear`` applies is done by the ``BitLinear`` leaf before the
dispatch, so the kernel itself never sees it (matching mlx-lm, which omits it; the
leaf adds it back for transformers fidelity). ``invert`` selects multiply (autobitlinear,
the ``microsoft/bitnet-b1.58-2B-4T`` checkpoint) versus divide.
"""

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_SOURCE = """
    constexpr int M = 4;
    constexpr int BLOCK = 32;

    uint tid = thread_position_in_grid.y;
    uint in_offset = thread_position_in_grid.x;

    uint out_packs = out_features / 4;
    uint batch_idx = tid / out_packs;
    uint row_idx = tid % out_packs;

    float sum[4] = {0.0};

    for (uint i = in_offset * M; i < in_features; i += BLOCK * M) {
        float v[M];
        for (int j = 0; j < M; j++) {
            v[j] = x[batch_idx * in_features + i + j];
        }
        for (int j = 0; j < M; j++) {
            uint8_t w = packed_weights[row_idx * in_features + i + j];
            sum[0] += v[j] * ((w & 3) - 1);
            sum[1] += v[j] * (((w >> 2) & 3) - 1);
            sum[2] += v[j] * (((w >> 4) & 3) - 1);
            sum[3] += v[j] * (((w >> 6) & 3) - 1);
        }
    }

    for (int j = 0; j < 4; j++) {
        sum[j] = simd_sum(sum[j]);
    }

    if (in_offset == 0) {
        float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
        for (int i = 0; i < 4; i++) {
            out[batch_idx * out_features + row_idx + i * out_packs] =
                static_cast<T>(sum[i] * scale);
        }
    }
"""

_KERNEL = metal_kernel(
    name="bitlinear_matmul",
    input_names=["x", "packed_weights", "weight_scale"],
    output_names=["out"],
    source=_SOURCE,
)


def bitlinear_applies(out_features: int, in_features: int, weight: mx.array) -> bool:
    """The reduction tiles cleanly: 4 outputs per byte, the 32-lane split-K steps in
    blocks of ``32 * 4 = 128`` input columns, and the weight is the checkpoint's
    packed uint8 table. Bit width and batch never disqualify."""
    return (
        weight.dtype == mx.uint8
        and out_features % 4 == 0
        and in_features % 128 == 0
        and weight.shape == (out_features // 4, in_features)
    )


def bitlinear(
    x: mx.array, weight: mx.array, weight_scale: mx.array, *, invert: bool = False
) -> mx.array:
    """``x`` ``[N, in]`` through the packed ternary ``weight`` ``[out//4, in]``,
    scaled by the scalar ``weight_scale`` -> ``[N, out]``.

    ``x`` and ``weight_scale`` must share a dtype; that dtype is the kernel's ``T``.
    """
    n = x.shape[0]
    in_features = x.shape[1]
    out_features = weight.shape[0] * 4
    assert bitlinear_applies(out_features, in_features, weight)
    assert x.dtype == weight_scale.dtype, "input and weight_scale dtypes must match"
    return _KERNEL(
        inputs=[x, weight, weight_scale],
        template=[
            ("T", weight_scale.dtype),
            ("invert_weight_scales", 1 if invert else 0),
            ("in_features", in_features),
            ("out_features", out_features),
        ],
        grid=(32, n * out_features // 4, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n, out_features)],
        output_dtypes=[weight_scale.dtype],
    )[0]
