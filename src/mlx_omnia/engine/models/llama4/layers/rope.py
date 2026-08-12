import mlx.core as mx

from mlx_omnia.engine.models.llama4.config import Llama4RoPEParameters


def llama3_rope(head_dim: int, base: float, scaling: Llama4RoPEParameters) -> mx.array:
    """The NTK-by-parts frequency table for llama3 RoPE scaling. Matches the reference
    implementation and transformers' `_compute_llama3_parameters` bit-for-bit in fp32.

    `freqs` is the actual frequency `base^(2i/d)` (mlx's `mx.fast.rope` computes
    `angle = offset / freqs`), not `inv_freq = 1/freqs`.
    """
    factor = scaling.factor
    low_freq_factor = scaling.low_freq_factor
    high_freq_factor = scaling.high_freq_factor
    old_context_len = scaling.original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    freqs = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    wavelens = 2 * mx.pi * freqs

    freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
    if high_freq_factor == low_freq_factor:
        return freqs
    is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
    smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
    return mx.where(is_medium_freq, smooth_freqs, freqs)
