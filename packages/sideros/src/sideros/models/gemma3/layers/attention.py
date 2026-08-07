import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.core.masks import SLIDING, causal_mask
from sideros.models.gemma3.config import Gemma3TextConfig


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding = layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self.rope_base = config.rope_local_base_freq if self.sliding else config.rope_theta
        self.scale = config.query_pre_attn_scalar**-0.5
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """qkv split back into per-head [1, heads, length, head_dim], normed but
        unrotated — the boundary transformers' q_norm/k_norm hooks expose."""
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        return self.q_norm(q), self.k_norm(k), v

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | None:
        """A lone query attends to everything it can reach; only a sliding layer whose
        cache already exceeds the window still needs a mask at T=1."""
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = self.split_heads(x)
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset),
            keys,
            values,
            scale=self.scale,
            mask=self.mask(length, keys.shape[2]),
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
