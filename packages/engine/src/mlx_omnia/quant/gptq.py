"""GPTQ (arXiv:2210.17323) over the second moment the calibration already collected.

The method is layerwise and consumes `SecondMoment.statistics()` — `X^T X` averaged over
rows — never the activations again. All of its arithmetic is fp32: the Hessian, the
Cholesky, the error propagation and the reconstruction error.

The grid is mlx's affine grid, and it is obtained from `mx.quantize` itself over the
columns of the group at the moment the group opens, so no parameter search is
reimplemented here. What cannot go through `mx.quantize` is the *packing*: it would
recompute scales and biases from the already compensated columns, and those differ from
the ones the codes were rounded against (a column is compensated after its group's
parameters are fixed). The codes therefore go through the format's own `pack_affine` and
are handed to `AffineWeight` directly.
"""

from dataclasses import dataclass

import mlx.core as mx

from mlx_omnia.quant.quantization import Affine, AffineWeight, pack_affine

_CPU = mx.Device(mx.cpu)
"""The three factorizations below have no Metal implementation in mlx 0.30."""


@dataclass(frozen=True)
class GPTQConfig:
    """`damping` is a fraction of the mean diagonal of the Hessian, so it is invariant to
    how the second moment was averaged. `block_size` is a multiple of the group size: a
    group straddling two blocks would read columns the current block has not compensated
    yet."""

    damping: float = 0.01
    block_size: int = 128
    act_order: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.damping < 1.0:
            raise ValueError(f"damping must sit in [0, 1), got {self.damping}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


_DEFAULT_CONFIG = GPTQConfig()


@dataclass(frozen=True)
class PermutedAffineWeight:
    """Codes in the checkpoint's column order plus, per column, the group that owns its
    scale. No runtime reads this: every affine kernel takes a group as a contiguous run of
    columns, and with activation ordering the runs are interleaved. It exists so the
    permutation survives as data instead of being dropped to fit the affine path."""

    codes: mx.array
    scales: mx.array
    biases: mx.array
    g_idx: mx.array
    format: Affine

    def __post_init__(self) -> None:
        if self.codes.dtype != mx.uint8:
            raise ValueError(f"codes must have dtype uint8, got {self.codes.dtype}")
        if self.codes.ndim != 2:
            raise ValueError(f"codes must be two-dimensional, got {self.codes.ndim}")
        rows, columns = self.codes.shape
        groups = columns // self.format.group_size
        if self.scales.shape != (rows, groups) or self.biases.shape != (rows, groups):
            raise ValueError(f"expected scales and biases of shape {(rows, groups)}")
        if self.g_idx.shape != (columns,):
            raise ValueError(f"expected a g_idx of shape {(columns,)}, got {self.g_idx.shape}")

    def dequantize(self) -> mx.array:
        scales = self.scales[:, self.g_idx].astype(mx.float32)
        biases = self.biases[:, self.g_idx].astype(mx.float32)
        return self.codes.astype(mx.float32) * scales + biases


def to_affine(weight: AffineWeight | PermutedAffineWeight) -> AffineWeight:
    """The executable path. It converts only when the group of every column is the run the
    affine layout implies; otherwise it stops, because reordering the codes to make them
    contiguous changes which weights share a scale."""
    if isinstance(weight, AffineWeight):
        return weight
    if 32 % weight.format.bits:
        raise ValueError(
            f"{weight.format.bits}-bit codes do not fill a uint32 word evenly; only 2, 4 "
            "and 8 have a GPTQ layout verified against mx.quantize"
        )
    columns = weight.codes.shape[-1]
    contiguous = mx.arange(columns) // weight.format.group_size
    if not mx.array_equal(weight.g_idx, contiguous).item():
        raise ValueError(
            "the result carries an explicit g_idx: its groups are not contiguous runs of "
            "columns, so it has no affine representation and needs its own runtime"
        )
    return AffineWeight(
        pack_affine(weight.codes, weight.format.bits),
        weight.scales,
        weight.biases,
        weight.format,
    )


def reconstruction_error(
    weight: mx.array, approximation: mx.array, second_moment: mx.array
) -> float:
    """The residual in the norm the layerwise objective uses: sqrt(tr(E H E^T)) with
    H = 2 X^T X, which is 2 ||X E^T||_F^2 under the square root. Not the plain Frobenius
    norm of E — the point of the method is that the input directions are not equally
    loaded."""
    error = (approximation.astype(mx.float32) - weight.astype(mx.float32)).reshape(
        -1, weight.shape[-1]
    )
    hessian = 2.0 * second_moment.astype(mx.float32)
    value = mx.sqrt(mx.sum((error @ hessian) * error)).item()
    assert isinstance(value, float)
    return value


@dataclass(frozen=True)
class GPTQSolution:
    """What the sequential pass produces, before any packing: the codes in the
    checkpoint's column order, the parameters of each group in the order the groups were
    opened, the value each column carried when it was rounded, and — under activation
    ordering — the group of each column."""

    codes: mx.array
    scales: mx.array
    biases: mx.array
    compensated: mx.array
    g_idx: mx.array


@dataclass(frozen=True)
class GPTQResult:
    weight: AffineWeight | PermutedAffineWeight
    error: float


def _hessian(second_moment: mx.array, columns: int, damping: float) -> tuple[mx.array, mx.array]:
    if second_moment.ndim != 2 or second_moment.shape != (columns, columns):
        raise ValueError(
            f"expected a second moment of shape {(columns, columns)}, got {second_moment.shape}"
        )
    hessian = 2.0 * second_moment.astype(mx.float32)
    diagonal = mx.diagonal(hessian)
    dead = diagonal <= 0
    hessian = hessian + mx.diag(mx.where(dead, mx.ones_like(diagonal), mx.zeros_like(diagonal)))
    mean_diagonal = mx.diagonal(hessian).mean()
    return hessian + mx.eye(columns) * (damping * mean_diagonal), dead


def _inverse_cholesky(hessian: mx.array) -> mx.array:
    """The upper Cholesky factor of H^-1: row i of it carries, scaled by its diagonal, the
    direction the error of column i propagates along the columns still to be quantized."""
    lower = mx.linalg.cholesky(hessian, stream=_CPU)
    inverse = mx.linalg.cholesky_inv(lower, stream=_CPU)
    factor = mx.linalg.cholesky(inverse, upper=True, stream=_CPU)
    mx.eval(factor)
    return factor


def solve(
    weight: mx.array,
    second_moment: mx.array,
    format: Affine,
    config: GPTQConfig = _DEFAULT_CONFIG,
) -> GPTQSolution:
    if weight.ndim != 2:
        raise ValueError(f"GPTQ takes a two-dimensional weight, got {weight.ndim} dimensions")
    if not mx.issubdtype(weight.dtype, mx.floating):
        raise ValueError(f"GPTQ requires a floating-point weight, got {weight.dtype}")
    rows, columns = weight.shape
    group = format.group_size
    if columns % group:
        raise ValueError(f"input dimension {columns} is not divisible by group size {group}")
    if config.block_size % group:
        raise ValueError(
            f"block_size {config.block_size} is not a multiple of group size {group}"
        )

    hessian, dead = _hessian(second_moment, columns, config.damping)
    order = (
        mx.argsort(-mx.diagonal(hessian)) if config.act_order else mx.arange(columns)
    )
    hessian = hessian[order][:, order]
    inverse = _inverse_cholesky(hessian)

    working = mx.zeros((rows, columns), mx.float32)
    working[:, :] = mx.where(dead[None, :], 0.0, weight.astype(mx.float32))[:, order]
    codes = mx.zeros((rows, columns), mx.uint8)
    compensated = mx.zeros((rows, columns), mx.float32)
    scales: list[mx.array] = []
    biases: list[mx.array] = []
    top = float((1 << format.bits) - 1)
    scale = mx.zeros((rows, 1), mx.float32)
    bias = mx.zeros((rows, 1), mx.float32)
    divisor = mx.ones((rows, 1), mx.float32)

    for start in range(0, columns, config.block_size):
        stop = min(start + config.block_size, columns)
        residual = mx.zeros((rows, stop - start), mx.float32)
        for index in range(start, stop):
            if index % group == 0:
                _, group_scale, group_bias = mx.quantize(
                    working[:, index : index + group].astype(weight.dtype),
                    group_size=group,
                    bits=format.bits,
                )
                scales.append(group_scale)
                biases.append(group_bias)
                scale = group_scale.astype(mx.float32)
                bias = group_bias.astype(mx.float32)
                divisor = mx.where(scale == 0, 1.0, scale)
            column = working[:, index : index + 1]
            code = mx.clip(mx.round((column - bias) / divisor), 0.0, top)
            error = (column - (code * scale + bias)) / inverse[index, index]
            codes[:, index : index + 1] = code.astype(mx.uint8)
            compensated[:, index : index + 1] = column
            residual[:, index - start : index - start + 1] = error
            if index + 1 < stop:
                working[:, index + 1 : stop] = working[:, index + 1 : stop] - (
                    error * inverse[index, index + 1 : stop][None, :]
                )
        if stop < columns:
            working[:, stop:] = working[:, stop:] - residual @ inverse[start:stop, stop:]
        mx.eval(working, codes, compensated)

    inverse_order = mx.argsort(order)
    g_idx = (mx.arange(columns) // group)[inverse_order]
    return GPTQSolution(
        codes=codes[:, inverse_order],
        scales=mx.concatenate(scales, axis=1),
        biases=mx.concatenate(biases, axis=1),
        compensated=compensated[:, inverse_order],
        g_idx=g_idx,
    )


def gptq(
    weight: mx.array,
    second_moment: mx.array,
    format: Affine,
    config: GPTQConfig = _DEFAULT_CONFIG,
) -> GPTQResult:
    """One layer, quantized against its own second moment. The result carries the residual
    in the Hessian norm so the gate can register the reconstruction error per layer."""
    solution = solve(weight, second_moment, format, config)
    permuted = PermutedAffineWeight(
        solution.codes, solution.scales, solution.biases, solution.g_idx, format
    )
    quantized: AffineWeight | PermutedAffineWeight
    if config.act_order:
        quantized = permuted
        approximation = permuted.dequantize()
    else:
        quantized = to_affine(permuted)
        approximation = quantized.dequantize()
    mx.eval(approximation)
    return GPTQResult(quantized, reconstruction_error(weight, approximation, second_moment))
