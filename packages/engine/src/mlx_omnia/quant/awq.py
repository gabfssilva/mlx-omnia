"""AWQ (arXiv:2306.00978) as what it is: a change of variables on the dense weights,
before any packing. A per-input-channel scale `s` multiplies the target's columns and
`1/s` is absorbed into the output channels of the module that produces its input, so
`(x / s) @ (s * W)ᵀ == x @ Wᵀ` exactly — the block's function is preserved *before* the
rounding, and what changes is only how the rounding error distributes over the channels.

Nothing here reaches the runtime: the result leaves as a dense weight dict, which
`quantize_weights` packs with the same `AffineRTN` as everything else. There is no AWQ
format, no AWQ mark in the checkpoint, and a leaf whose pair is refused is simply
quantized as it was.

The search is over `s = E[x²]^(alpha/2)`, alpha on a deterministic grid, scored by the
reconstruction error of the target's output under a diagonal input covariance — which is
exactly what the imatrix carries. The absorber's own quantization error is not in the
score: the transformation is chosen for the target, as in the paper.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.quant.calibration import quantize_dequantize
from mlx_omnia.quant.quantization import Affine, QuantizationPlan, inventory

ALPHA_GRID: tuple[float, ...] = tuple(step / 20 for step in range(21))

_FLOOR = 1e-12


@dataclass(frozen=True)
class Pair:
    """The target and the module whose output channels absorb the inverse scale. Only a
    weighted leaf feeding another weighted leaf is expressed here; a scale that would have
    to travel through a norm, a residual or a second consumer is a different pair and is
    refused rather than guessed."""

    target: str
    absorber: str


_GATED_MLP = frozenset({"gate_proj", "up_proj", "down_proj"})
_ATTENTION = frozenset({"v_proj", "o_proj"})


def derive_pairs(model: nn.Module) -> list[Pair]:
    """The pairs a tree admits *exactly*, read off the names its leaves already carry, and
    only the two that are exact:

    - `down_proj ← up_proj` when the three names of a gated MLP share a parent. The input of
      `down_proj` is `silu(gate) ⊙ up`, and a scale on the columns of `up` passes through the
      product untouched — the elementwise gate is a factor, not a function of the channel.
      A non-gated MLP is not this pair and is not derived: the activation between the two
      matrices is what breaks the identity, so GPT-2's `c_fc`/`c_proj` stays out.
    - `o_proj ← v_proj` when both share a parent. Attention is linear in the output channels
      of `v`, so a per-channel scale commutes with the softmax-weighted sum.

    Everything else — a scale that would have to cross a norm, a residual or a second
    consumer — is not derived here. A tree that admits neither pair gets an empty list, which
    is an AWQ run that applies nothing, not an error: that is every architecture in this
    package today, whose load-time fusions leave `qkv_proj` and `gate_up_proj` where the two
    absorbers would have been. Absorbing into half of an interleaved leaf is a different
    `Pair` than this one, and inventing it here would be inventing the exactness too.
    """
    children: dict[str, set[str]] = {}
    for leaf in inventory(model):
        parent, _, name = leaf.path.rpartition(".")
        children.setdefault(parent, set()).add(name)
    pairs: list[Pair] = []
    for parent in sorted(children):
        names = children[parent]
        prefix = f"{parent}." if parent else ""
        if names >= _GATED_MLP:
            pairs.append(Pair(f"{prefix}down_proj", f"{prefix}up_proj"))
        if names >= _ATTENTION:
            pairs.append(Pair(f"{prefix}o_proj", f"{prefix}v_proj"))
    return pairs


@dataclass(frozen=True)
class Search:
    alpha: float
    clip: float | None
    error: float


@dataclass(frozen=True)
class Applied:
    pair: Pair
    scale: mx.array
    search: Search
    rtn_error: float

    @property
    def improvement(self) -> float:
        return (self.rtn_error - self.search.error) / self.rtn_error


@dataclass(frozen=True)
class Refused:
    """A pair that does not admit the transformation. The leaf still gets quantized — by
    plain RTN — and the reason is carried out so the fallback is never silent."""

    pair: Pair
    reason: str


type Outcome = Applied | Refused


def channel_scale(mean_square: mx.array, alpha: float) -> mx.array:
    """`s = sqrt(E[x²])^alpha`, normalized by the geometric mean of its extremes so the
    magnitude of the scaled weight does not drift with alpha. At `alpha == 0` it is
    exactly one, which is what makes RTN a point of the grid instead of a separate path."""
    magnitude = mx.sqrt(mx.maximum(mean_square.astype(mx.float32), _FLOOR))
    scale = magnitude**alpha
    return scale / mx.sqrt(scale.max() * scale.min())


def _clipped(weight: mx.array, ratio: float) -> mx.array:
    limit = mx.abs(weight).max(axis=-1, keepdims=True) * ratio
    return mx.clip(weight, -limit, limit)


def _error(
    weight: mx.array,
    scale: mx.array,
    mean_square: mx.array,
    format: Affine,
    clip: float | None,
) -> tuple[float, mx.array]:
    """Expected squared error of the target's output under a diagonal input covariance:
    the input the scaled weight sees is `x / s`, so the channel weight is `E[x²] / s²`."""
    scaled = weight.astype(mx.float32) * scale
    if clip is not None:
        scaled = _clipped(scaled, clip)
    stored = scaled.astype(weight.dtype)
    residual = quantize_dequantize(stored, format).astype(mx.float32) - scaled
    weighting = mean_square.astype(mx.float32) / (scale * scale)
    value = (residual * residual * weighting).sum().item()
    assert isinstance(value, float)
    return value, stored


def _refuse(
    weights: Mapping[str, mx.array],
    plan: QuantizationPlan,
    pair: Pair,
    mean_square: mx.array | None,
) -> str | None:
    format = plan.get(pair.target)
    if format is None:
        return f"{pair.target} is not in the quantization plan"
    if not isinstance(format, Affine):
        return f"{pair.target} is planned as {format.mode}, which AWQ does not pack"
    if mean_square is None:
        return f"the calibration carries no imatrix statistic for {pair.target}"
    for path in (pair.target, pair.absorber):
        weight = weights.get(f"{path}.weight")
        if weight is None:
            return f"{path} is absent from the checkpoint"
        if weight.dtype == mx.uint32 or f"{path}.scales" in weights:
            return f"{path} is already quantized"
        if weight.ndim != 2:
            return f"{path} is not a two-dimensional weight"
    target, absorber = weights[f"{pair.target}.weight"], weights[f"{pair.absorber}.weight"]
    if absorber.shape[0] != target.shape[1]:
        return (
            f"{pair.absorber} produces {absorber.shape[0]} channels "
            f"and {pair.target} consumes {target.shape[1]}"
        )
    if mean_square.shape != (target.shape[1],):
        return (
            f"the imatrix of {pair.target} has shape {mean_square.shape}, "
            f"not ({target.shape[1]},)"
        )
    return None


def apply_awq(
    weights: dict[str, mx.array],
    plan: QuantizationPlan,
    statistics: Mapping[str, mx.array],
    pairs: Sequence[Pair],
    *,
    prefix: str = "imatrix/",
    alphas: Sequence[float] = ALPHA_GRID,
    clips: Sequence[float] = (),
) -> list[Outcome]:
    """Rewrites the dense weights in place and reports, pair by pair, what happened. The
    packing is not done here: the caller runs the same `quantize_weights` it would have
    run without AWQ."""
    outcomes: list[Outcome] = []
    consumed: dict[str, str] = {}
    for pair in pairs:
        owner = consumed.get(pair.absorber)
        if owner is not None:
            outcomes.append(
                Refused(pair, f"{pair.absorber} already absorbs the scale of {owner}")
            )
            continue
        mean_square = statistics.get(f"{prefix}{pair.target}.mean_square")
        reason = _refuse(weights, plan, pair, mean_square)
        if reason is not None:
            outcomes.append(Refused(pair, reason))
            continue
        assert mean_square is not None
        format = plan[pair.target]
        assert isinstance(format, Affine)

        target = weights[f"{pair.target}.weight"]
        candidates: list[tuple[float, float | None]] = [
            (alpha, clip) for alpha in alphas for clip in (None, *clips)
        ]
        rtn_error, _ = _error(target, channel_scale(mean_square, 0.0), mean_square, format, None)
        best = Search(0.0, None, rtn_error)
        best_weight = target
        best_scale = channel_scale(mean_square, 0.0)
        for alpha, clip in candidates:
            scale = channel_scale(mean_square, alpha)
            error, stored = _error(target, scale, mean_square, format, clip)
            if error < best.error:
                best, best_weight, best_scale = Search(alpha, clip, error), stored, scale

        absorber = weights[f"{pair.absorber}.weight"]
        inverse = (1.0 / best_scale).astype(absorber.dtype)
        weights[f"{pair.target}.weight"] = best_weight
        weights[f"{pair.absorber}.weight"] = absorber * inverse[:, None]
        absorber_bias = weights.get(f"{pair.absorber}.bias")
        if absorber_bias is not None:
            weights[f"{pair.absorber}.bias"] = absorber_bias * inverse
        mx.eval(
            weights[f"{pair.target}.weight"],
            weights[f"{pair.absorber}.weight"],
            best_scale,
        )
        consumed[pair.absorber] = pair.target
        outcomes.append(Applied(pair, best_scale, best, rtn_error))
    return outcomes
