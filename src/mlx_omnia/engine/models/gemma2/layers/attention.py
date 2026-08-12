import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.layers import split_qkv
from mlx_omnia.engine.core.masks import SLIDING, causal_mask
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.gemma2.config import Gemma2Config


def softcap(x: mx.array, cap: float) -> mx.array:
    return mx.tanh(x / cap) * cap


class Gemma2Attention(nn.Module):
    def __init__(self, config: Gemma2Config, layer_type: str) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.repeats = self.heads // self.kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.scale = config.query_pre_attn_scalar**-0.5
        self.cap = config.attn_cap
        self.window = config.sliding_window if layer_type == SLIDING else None
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = split_qkv(
            self.qkv_proj(x), heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim
        )
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        queries = self.rope(q, offset) * self.scale
        if self.repeats > 1:
            queries = queries.reshape(1, self.kv_heads, self.repeats, length, self.head_dim)
            keys = mx.expand_dims(keys, 2)
            values = mx.expand_dims(values, 2)
        scores = softcap(queries @ keys.swapaxes(-1, -2), self.cap)
        allowed = self.mask(length, keys.shape[-2])
        if allowed is not None:
            scores = mx.where(allowed, scores, mx.finfo(scores.dtype).min)
        attended = softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype) @ values
        if self.repeats > 1:
            attended = attended.reshape(1, self.heads, length, self.head_dim)
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
