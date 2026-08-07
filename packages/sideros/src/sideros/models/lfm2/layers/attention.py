import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv


class LFM2Attention(nn.Module):
    def __init__(
        self, hidden: int, *, heads: int, kv_heads: int, eps: float, rope_theta: float
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.rope_theta = rope_theta
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.out_proj = nn.Linear(queries, hidden, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """q/k rms-normed per head between the projection and the rotation, as in Qwen3."""
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        return self.q_layernorm(q), self.k_layernorm(k), v

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = self.split_heads(x)
        queries = self.rope(q, offset)
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.out_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))
