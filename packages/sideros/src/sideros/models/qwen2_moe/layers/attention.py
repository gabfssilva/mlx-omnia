import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.models.qwen2_moe.config import Qwen2MoEConfig


class Qwen2MoEAttention(nn.Module):
    def __init__(self, config: Qwen2MoEConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=True)
        self.o_proj = nn.Linear(queries, hidden, bias=False)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = split_qkv(
            self.qkv_proj(x), heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim
        )
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset), keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
