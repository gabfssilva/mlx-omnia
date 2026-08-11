"""Ternary matmul over uint8-packed 1.58-bit weights, ported from the
reference implementation.

The weight is packed 4 ternary values per byte along the output axis:
``packed[p, c]`` holds the four outputs ``p, p + out/4, p + 2*out/4, p + 3*out/4``
at input column ``c`` (bits 0-1, 2-3, 4-5, 6-7, LSB-first, field ``- 1`` -> ``{-1,0,1}``).
One threadgroup owns one (output-pack, batch-row); its 32 lanes split the
``in_features`` reduction (``BLOCK * M = 128`` columns per super-step) and close it
with ``simd_sum``; lane 0 writes the four scaled outputs.

The kernel is a pure float matmul over the unpacked ternary weights followed by a
single scalar ``weight_scale``. The per-token int8 activation fake-quant that
transformers' ``AutoBitLinear`` applies is done by the ``BitLinear`` leaf before the
dispatch, so the kernel itself never sees it (matching the reference, which omits it;
the leaf adds it back for transformers fidelity). ``invert`` selects multiply (autobitlinear,
the ``microsoft/bitnet-b1.58-2B-4T`` checkpoint) versus divide.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.core.kernels.qmv.kernel import Epilogue, QmvLeaf, TernaryLinear, is_ternary
from mlx_omnia.core.mxcompat import metal_kernel

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


def unpack_ternary(weight: mx.array) -> mx.array:
    """``packed [out//4, in]`` uint8 -> ``[out, in]`` float in the logical output order
    transformers' ``unpack_weights`` produces: field ``j`` of byte ``p`` is row
    ``p + j * (out//4)``."""
    out_packs = weight.shape[0]
    in_features = weight.shape[1]
    three = mx.array(3, dtype=mx.uint8)
    fields = [
        (weight & three),
        ((weight >> mx.array(2, dtype=mx.uint8)) & three),
        ((weight >> mx.array(4, dtype=mx.uint8)) & three),
        ((weight >> mx.array(6, dtype=mx.uint8)) & three),
    ]
    stacked = mx.concatenate(fields, axis=0)
    return stacked.astype(mx.float32).reshape(out_packs * 4, in_features) - mx.array(
        1.0, dtype=mx.float32
    )


def ternary_matmul(x: mx.array, weight: mx.array, weight_scale: mx.array) -> mx.array:
    """The op chain the kernel replaces: unpack to float, matmul, scale."""
    w = unpack_ternary(weight).astype(x.dtype)
    return (x @ w.T) * weight_scale


@dataclass(frozen=True)
class TernaryQmv:
    leaf: TernaryLinear
    in_features: int
    out_features: int

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
        if epilogue != "none" or not is_ternary(leaf):
            return None
        if (leaf.in_features, leaf.out_features) != (kdim, rows):
            return None
        if not bitlinear_applies(rows, kdim, leaf.weight):
            return None
        return cls(leaf, kdim, rows)

    def __call__(self, x: mx.array, gate: mx.array | None = None) -> mx.array:
        rows = x.reshape(-1, self.in_features)
        # Read through the leaf, not a snapshot: a swapped scale (mutation test,
        # quantize-on-load) must reach the kernel without a re-resolution.
        out = bitlinear(rows, self.leaf.weight, self.leaf.weight_scale)
        return out.reshape(*x.shape[:-1], self.out_features)
