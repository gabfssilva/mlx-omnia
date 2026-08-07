"""RoPE period tables for the scaling formulas more than one architecture declares.

`mx.fast.rope(freqs=)` consumes periods, not inverse frequencies; both builders below
return the `[dims/2]` period table in that convention.
"""

import math
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import mlx.core as mx


@dataclass(frozen=True)
class LlamaRoPEScaling:
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


class ScalingJson(TypedDict):
    rope_type: str
    factor: float
    low_freq_factor: NotRequired[float]
    high_freq_factor: NotRequired[float]
    original_max_position_embeddings: NotRequired[int]


def llama3_freqs(head_dim: int, base: float, scaling: LlamaRoPEScaling) -> mx.array:
    """The `[head_dim/2]` period table `mx.fast.rope(freqs=)` consumes.

    transformers works on `inv_freq = 1/period` and *divides* by the factor
    (`modeling_rope_utils.py`, `_compute_llama3_parameters`); the period convention
    multiplies. Same op order as the reference implementation, so the table matches it bit for
    bit. `attention_factor` is 1.0 in this formula — nothing pre-scales q/k.
    """
    low = scaling.original_max_position_embeddings / scaling.low_freq_factor
    high = scaling.original_max_position_embeddings / scaling.high_freq_factor

    periods = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    wavelengths = 2 * math.pi * periods

    stretched = mx.where(wavelengths > low, periods * scaling.factor, periods)
    medium = (wavelengths > high) & (wavelengths < low)
    smooth = (scaling.original_max_position_embeddings / wavelengths - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    interpolated = stretched / ((1 - smooth) / scaling.factor + smooth)
    return mx.where(medium, interpolated, stretched)


def llama3_scaling(raw: ScalingJson | None) -> LlamaRoPEScaling | None:
    """Only the `llama3` formula. A checkpoint declaring `linear`, `dynamic` or `yarn`
    under `model_type: "llama"` would otherwise load and rotate at the wrong periods."""
    if raw is None:
        return None
    if raw["rope_type"] != "llama3":
        raise ValueError(f"unsupported llama rope_type {raw['rope_type']!r}")
    return LlamaRoPEScaling(
        factor=raw["factor"],
        low_freq_factor=raw.get("low_freq_factor", 1.0),
        high_freq_factor=raw.get("high_freq_factor", 4.0),
        original_max_position_embeddings=raw.get("original_max_position_embeddings", 8192),
    )


class YarnJson(TypedDict):
    factor: float
    original_max_position_embeddings: NotRequired[int]
    beta_fast: NotRequired[float]
    beta_slow: NotRequired[float]
    mscale: NotRequired[float]
    mscale_all_dim: NotRequired[float]
    type: NotRequired[str]
    rope_type: NotRequired[str]


@dataclass(frozen=True)
class Yarn:
    """The rotary table and the two places its `mscale` is used."""

    freqs: mx.array | None
    mscale: float
    scale_correction: float


def yarn_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn(dims: int, base: float, scaling: YarnJson | None) -> Yarn:
    """NTK-by-parts: the interpolated and the extrapolated period tables blended by a
    linear ramp over the dimensions whose wavelength falls between `beta_fast` and
    `beta_slow` rotations at the original context length."""
    if scaling is None:
        return Yarn(None, 1.0, 1.0)
    factor = scaling["factor"]
    positions = scaling.get("original_max_position_embeddings", 4096)
    beta_fast = scaling.get("beta_fast", 32)
    beta_slow = scaling.get("beta_slow", 1)
    mscale = scaling.get("mscale", 1)
    mscale_all_dim = scaling.get("mscale_all_dim", 0)

    def correction(rotations: float) -> float:
        return (dims * math.log(positions / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), dims - 1)
    if low == high:
        high += 0.001
    extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    inter = factor * extra
    ramp = mx.clip((mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    correction_scale = (
        yarn_mscale(factor, mscale_all_dim) ** 2 if mscale_all_dim else 1.0
    )
    return Yarn(
        freqs=(inter * extra) / (inter * mask + extra * (1 - mask)),
        mscale=yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim),
        scale_correction=correction_scale,
    )


