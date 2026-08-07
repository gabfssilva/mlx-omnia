import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.models.step3p7.config import SLIDING, Step3p7RoPEScaling, Step3p7TextConfig


class Step3p7Attention(nn.Module):
    def __init__(self, config: Step3p7TextConfig, layer: int) -> None:
        super().__init__()
        self.heads = config.heads_per_layer[layer]
        self.kv_heads = config.num_attention_groups
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False)

        layer_type = config.types[layer]
        self.sliding = layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self._rotary_dim = int(self.head_dim * config.rotary_factors[layer])
        self._theta = config.thetas[layer]

        scaling = config.rope_scaling
        if scaling is not None and layer_type in config.yarn_only_types:
            self._freqs = _llama3_freqs(self._rotary_dim, self._theta, scaling)
            mx.eval(self._freqs)
        else:
            self._freqs = None

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        queries = self._rope(self.q_norm(q), offset)
        rotated = self._rope(self.k_norm(k), offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, query_width)
        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = (
            output.reshape(1, length, self.heads, self.head_dim) * gate[..., None]
        ).reshape(1, length, query_width)
        return self.o_proj(output)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        if self._freqs is not None:
            return mx.fast.rope(
                x, self._rotary_dim, traditional=False, base=None, scale=1.0,
                offset=offset, freqs=self._freqs,
            )
        return mx.fast.rope(
            x, self._rotary_dim, traditional=False, base=self._theta, scale=1.0,
            offset=offset,
        )


def _llama3_freqs(rotary_dim: int, base: float, scaling: Step3p7RoPEScaling) -> mx.array:
    factor = scaling.factor
    original_max = scaling.original_max_position_embeddings
    low_freq_wavelen = original_max / scaling.low_freq_factor

    exponents = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim
    freqs = base**exponents
    wavelens = (2 * math.pi) / freqs

    t = (original_max / wavelens - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    smooth = 0.5 * (1.0 - mx.cos(mx.pi * mx.clip(t, 0.0, 1.0)))
    scaled = freqs * (1.0 - smooth) / factor + freqs * smooth
    return mx.where(wavelens > low_freq_wavelen, freqs / factor, scaled)
