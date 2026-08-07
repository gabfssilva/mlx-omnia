import math

import mlx.core as mx

from sideros.models.gpt_oss.config import GPTOSSRoPEScaling


def yarn_rope(head_dim: int, base: float, scaling: GPTOSSRoPEScaling) -> tuple[mx.array, float]:
    """The NTK-by-parts frequency table and the length scale applied to q/k before the
    rotation. Same op order as the reference implementation, so the table matches bit for bit."""
    factor = scaling.factor
    original = scaling.original_max_position_embeddings

    def correction(rotations: float) -> float:
        return (head_dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(scaling.beta_fast)), 0)
    high = min(math.ceil(correction(scaling.beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001

    extra = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    inter = factor * extra
    ramp = mx.clip((mx.arange(head_dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
    # yarn_get_mscale(factor, mscale=1) / yarn_get_mscale(factor, mscale_all_dim=0); the
    # denominator is 1 whenever the checkpoint omits mscale_all_dim, as gpt-oss does.
    mscale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return freqs, mscale
