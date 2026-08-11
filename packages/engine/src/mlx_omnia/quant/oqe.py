"""oQe: the grid of each group chosen against the imatrix, not by the extremes alone.

RTN reads scale and bias off the group's minimum and maximum, which spends the whole grid
on whatever the outliers of the group are, whether or not the channels they sit in ever
carry signal. oQe scores a small set of candidate grids by the error each causes *weighted
by how much the input channel is used* — `E[x²]` per channel, the imatrix the calibration
already collects and AWQ already reads — and keeps, per group, the one that wins. A
channel nobody activated does not vote on where the grid goes.

The candidates are one family. A grid is anchored at one of the group's extremes and
aligned to zero: the anchor sits at an integer number of steps from zero, so both the
extreme and zero are exactly representable, and what is swept is the step — below the
min/max step it clips the opposite tail, above it widens. Two anchors, seven factors, and
the min/max-magnitude anchor at factor one seeds the search, which is the grid RTN itself
would have kept. `mx.quantize`'s own parameters are not among the candidates: they are the
un-aligned version of that seed, and a lattice that misses zero rounds every small weight
of the group against a half-step offset.

The scales come out negative wherever the anchor is the maximum — the codes then run
downward from it — and that is deliberate rather than a normalization worth doing:
`bias == edge` is a value the checkpoint already carries and stores back exactly, while the
mirrored form would round the anchor away at bf16. Nothing downstream reads the sign: an
affine grid is `code * scale + bias` in `mx.dequantize`, in `quantized_matmul` and in the
package's own kernels.

This is the packing alone. Which leaf gets how many bits is `oq.py`'s allocator, and the
two compose without knowing each other: what comes out here is an ordinary `AffineWeight`,
in the layout `mx.quantize` produces, and no loader can tell which method rounded it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_omnia.quant.quantization import Affine, AffineWeight, pack_affine

FACTORS: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25)
"""The step of the anchored grid, as a fraction of the min/max step."""

_STEP_FLOOR = 1e-7
"""A constant group has no range of its own. Its grid is the anchor plus a step nothing
reaches, which reconstructs the constant exactly — where `mx.quantize` gets a zero scale."""


def _aligned(edge: mx.array, step: mx.array) -> tuple[mx.array, mx.array]:
    """The grid of that step which has `edge` on it and is aligned to zero: `edge` is
    `round(edge / step)` steps away from the origin, and the step is stretched to close that
    rounding. Under one step away there is no such integer, and what is left is the grid
    through zero with the step it was given."""
    index = mx.round(edge / step)
    return mx.where(index != 0, edge / index, step), mx.where(index != 0, edge, 0.0)


@dataclass(frozen=True)
class ImportanceMatrixAffine:
    """One leaf's weighted quantizer: the imatrix of *this* leaf, one number per input
    channel. It is a `Method[Affine, AffineWeight]`, so the caller is the same
    `quantize_weights` the RTN path runs — with the search in place of the rounding."""

    mean_square: mx.array
    factors: Sequence[float] = field(default=FACTORS)

    def quantize(self, weight: mx.array, format: Affine) -> AffineWeight:
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"oQe requires a floating-point weight, got {weight.dtype}")
        if weight.ndim < 2:
            raise ValueError(f"weight must have at least two dimensions, got {weight.ndim}")
        columns = weight.shape[-1]
        if columns % format.group_size:
            raise ValueError(
                f"input dimension {columns} is not divisible by group size "
                f"{format.group_size}"
            )
        if self.mean_square.shape != (columns,):
            raise ValueError(
                f"the imatrix has shape {self.mean_square.shape}, not ({columns},)"
            )

        groups = columns // format.group_size
        top = float((1 << format.bits) - 1)
        # The rows are flattened: a stacked expert bank is quantized like any other matrix,
        # and the imatrix — one vector over the input channels — is the same for all of them.
        grouped = weight.astype(mx.float32).reshape(-1, groups, format.group_size)
        importance = self.mean_square.astype(mx.float32).reshape(
            1, groups, format.group_size
        )
        low = grouped.min(axis=-1, keepdims=True)
        high = grouped.max(axis=-1, keepdims=True)
        step = mx.maximum((high - low) / top, _STEP_FLOOR)

        def codes_of(scales: mx.array, biases: mx.array) -> mx.array:
            return mx.clip(mx.round((grouped - biases) / scales), 0.0, top)

        def error(scales: mx.array, biases: mx.array) -> mx.array:
            residual = grouped - (codes_of(scales, biases) * scales + biases)
            return (importance * residual * residual).sum(axis=-1, keepdims=True)

        outer = mx.abs(low) > mx.abs(high)
        best_scales, best_biases = _aligned(
            mx.where(outer, low, high), mx.where(outer, step, -step)
        )
        best_error = error(best_scales, best_biases)
        for anchor, direction in ((high, -1.0), (low, 1.0)):
            scale, bias = _aligned(anchor, step * direction)
            for factor in self.factors:
                scales = scale * factor
                candidate = error(scales, bias)
                taken = candidate < best_error
                best_error = mx.where(taken, candidate, best_error)
                best_scales = mx.where(taken, scales, best_scales)
                best_biases = mx.where(taken, bias, best_biases)
            mx.eval(best_error, best_scales, best_biases)

        codes = codes_of(best_scales, best_biases).reshape(*weight.shape[:-1], columns)
        shape = (*weight.shape[:-1], groups)
        return AffineWeight(
            pack_affine(codes, format.bits),
            best_scales.reshape(shape).astype(weight.dtype),
            best_biases.reshape(shape).astype(weight.dtype),
            format,
        )
