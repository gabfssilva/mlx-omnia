import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.olmoe.config import OlmoEConfig


class OlmoEAttention(nn.Module):
    def __init__(self, config: OlmoEConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        self.queries = self.heads * self.head_dim
        self.key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, self.queries + 2 * self.key_values, bias=False)
        self.o_proj = nn.Linear(self.queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.queries, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.key_values, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        flat = self.qkv_proj(x)
        q, k, v = mx.split(flat, [self.queries, self.queries + self.key_values], axis=-1)
        heads = [
            part.reshape(1, length, count, self.head_dim).transpose(0, 2, 1, 3)
            for part, count in (
                (self.q_norm(q), self.heads),
                (self.k_norm(k), self.kv_heads),
                (v, self.kv_heads),
            )
        ]
        keys, values = cache.update_and_fetch(self.rope(heads[1], offset), heads[2])
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(heads[0], offset), keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, self.queries))
