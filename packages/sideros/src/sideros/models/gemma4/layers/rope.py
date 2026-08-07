import mlx.core as mx


def proportional_inv_freq(head_dim: int, partial: float, theta: float) -> mx.array:
    """The `inv_freq` vector for proportional partial-rotary RoPE, in fp32.

    `int(partial * head_dim // 2)` real frequency pairs computed with the exponent
    denominator = `head_dim` (not the rotated count), followed by zero-frequency
    pairs (identity: cos=1, sin=0). Matches transformers'
    `_compute_proportional_rope_parameters`.
    """
    rope_angles = int(partial * head_dim // 2)
    real = mx.power(
        theta,
        mx.arange(0, rope_angles, dtype=mx.float32) * (-2.0 / head_dim),
    )
    zeros = mx.zeros(head_dim // 2 - rope_angles, dtype=mx.float32)
    return mx.concatenate([real, zeros])


def cos_sin_tables(
    inv_freq: mx.array, positions: mx.array, head_dim: int
) -> tuple[mx.array, mx.array]:
    """Precompute fp32 cos/sin tables for manual RoPE.

    `positions` is `[length]`; returns `[length, head_dim//2]` each.
    """
    freqs = mx.outer(positions.astype(mx.float32), inv_freq)
    # [length, head_dim//2] — freqs for each position, each frequency pair
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    return cos, sin


def manual_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE manually to `x` of shape `[1, heads, length, head_dim]`.

    Split-half (transformers `rotate_half` with `emb = cat(freqs, freqs)`): dim `j`
    pairs with `j + head_dim//2` and both share frequency `j`. The zero-frequency
    inv_freq tail makes the nope dims identity (cos=1, sin=0), so `cos`/`sin` span
    the full `head_dim//2`.
    """
    half = cos.shape[-1]
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin are [length, head_dim//2] — broadcast over heads.
    cos_b = cos[mx.newaxis, mx.newaxis, :, :]
    sin_b = sin[mx.newaxis, mx.newaxis, :, :]
    return mx.concatenate([x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], axis=-1)
