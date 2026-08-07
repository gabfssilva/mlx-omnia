import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.models.phi3.config import Phi3Config


class Phi3Attention(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.rope_dims = config.rope_dims
        self.rope_theta = config.rope_theta
        self.rope_scale = config.rope_scale
        self.long_rope = config.long_rope
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        if self.long_rope is None:
            return mx.fast.rope(
                x,
                self.rope_dims,
                traditional=False,
                base=self.rope_theta,
                scale=self.rope_scale,
                offset=offset,
            )
        return mx.fast.rope(
            x * self.long_rope.scale,
            self.rope_dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self.long_rope.freqs,
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
            scale=self.head_dim**-0.5,
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
