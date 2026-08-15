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

    `positions` is `[length]` for one sequence, or `[batch, length]` when the rows of a
    ragged batch stand at different offsets; the tables come back `[length, head_dim//2]`
    or `[batch, 1, length, head_dim//2]`, ready to broadcast over heads either way.
    """
    if positions.ndim == 1:
        freqs = mx.outer(positions.astype(mx.float32), inv_freq)
    else:
        freqs = positions.astype(mx.float32)[:, mx.newaxis, :, mx.newaxis] * inv_freq
    return mx.cos(freqs), mx.sin(freqs)


def manual_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE manually to `x` of shape `[batch, heads, length, head_dim]`.

    Split-half (transformers `rotate_half` with `emb = cat(freqs, freqs)`): dim `j`
    pairs with `j + head_dim//2` and both share frequency `j`. The zero-frequency
    inv_freq tail makes the nope dims identity (cos=1, sin=0), so `cos`/`sin` span
    the full `head_dim//2`.
    """
    half = cos.shape[-1]
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_b = cos if cos.ndim == 4 else cos[mx.newaxis, mx.newaxis]
    sin_b = sin if sin.ndim == 4 else sin[mx.newaxis, mx.newaxis]
    return mx.concatenate([x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], axis=-1)
