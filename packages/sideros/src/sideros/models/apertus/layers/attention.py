import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.core.rope import llama3_freqs
from sideros.models.apertus.config import ApertusConfig


class ApertusAttention(nn.Module):
    def __init__(self, config: ApertusConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        scaling = config.scaling
        self._freqs = (
            None
            if scaling is None
            else llama3_freqs(self.head_dim, config.rope_theta, scaling)
        )
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        if self._freqs is None:
            return mx.fast.rope(
                x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
            )
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=None, scale=1.0, offset=offset,
            freqs=self._freqs,
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = split_qkv(
            self.qkv_proj(x), heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim
        )
        keys, values = cache.update_and_fetch(self.rope(self.k_norm(k), offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(self.q_norm(q), offset), keys, values,
            scale=self.head_dim**-0.5,
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
