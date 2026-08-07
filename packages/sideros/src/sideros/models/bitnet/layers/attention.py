import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.bitnet.config import BitNetConfig
from sideros.models.bitnet.layers.bitlinear import BitLinear


class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.q_proj = BitLinear(hidden, queries)
        self.k_proj = BitLinear(hidden, key_values)
        self.v_proj = BitLinear(hidden, key_values)
        self.o_proj = BitLinear(queries, hidden)
        self.attn_sub_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        # Split-half (transformers rotate_half), not the interleaved convention.
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q = self.q_proj(x).reshape(1, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries, rotated = self.rope(q, offset), self.rope(k, offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        out = attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        return self.o_proj(self.attn_sub_norm(out))
