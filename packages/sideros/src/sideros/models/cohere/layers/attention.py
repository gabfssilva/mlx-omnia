import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.cohere.config import CohereConfig


class HeadLayerNorm(nn.Module):
    """LayerNorm over the head dimension with a `[heads, head_dim]` weight — the shape
    the checkpoint stores, and the reason this is not `nn.LayerNorm`."""

    def __init__(self, heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((heads, head_dim))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps)


class CohereAttention(nn.Module):
    def __init__(self, config: CohereConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        self.queries = self.heads * self.head_dim
        self.key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            hidden, self.queries + 2 * self.key_values, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.queries, hidden, bias=config.attention_bias)
        if config.use_qk_norm:
            self.q_norm = HeadLayerNorm(self.heads, self.head_dim, config.layer_norm_eps)
            self.k_norm = HeadLayerNorm(self.kv_heads, self.head_dim, config.layer_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        flat = self.qkv_proj(x)
        q, k, v = mx.split(flat, [self.queries, self.queries + self.key_values], axis=-1)
        q = q.reshape(1, length, self.heads, self.head_dim)
        k = k.reshape(1, length, self.kv_heads, self.head_dim)
        if "q_norm" in self:
            q, k = self.q_norm(q), self.k_norm(k)
        keys, values = cache.update_and_fetch(
            self.rope(k.transpose(0, 2, 1, 3), offset),
            v.reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3),
        )
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q.transpose(0, 2, 1, 3), offset), keys, values,
            scale=self.head_dim**-0.5,
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, self.queries))
