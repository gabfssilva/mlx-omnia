import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.models.falcon_h1.config import FalconH1Config


class FalconH1Attention(nn.Module):
    """GQA + plain RoPE (base 1e11, full rotary, no scaling). The
    ``key_multiplier`` is folded into ``k_proj.weight`` at load (RoPE is linear,
    so scaling k before or after rotation is the same)."""

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, query_width + 2 * kv_width, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(query_width, config.hidden_size, bias=config.attention_bias)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        offset = cache.offset
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        q = mx.fast.rope(
            q, self.head_dim, traditional=False, base=config.rope_theta, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, self.head_dim, traditional=False, base=config.rope_theta, scale=1.0, offset=offset
        )
        keys, values = cache.update_and_fetch(k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale,
            mask=None if length == 1 else "causal",
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        return self.o_proj(output)
