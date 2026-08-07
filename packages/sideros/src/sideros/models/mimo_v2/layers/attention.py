import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.masks import causal_mask
from sideros.models.mimo_v2.config import SLIDING, LayerType, MimoV2Config


class MimoV2Attention(nn.Module):
    def __init__(self, config: MimoV2Config, layer_type: LayerType) -> None:
        super().__init__()
        sliding = layer_type == SLIDING
        self.heads = config.num_attention_heads
        # The sliding layers carry twice the kv heads the full-attention ones do.
        self.kv_heads = config.num_key_value_heads * (2 if sliding else 1)
        self.head_dim = config.head_dim
        self.v_head_dim = config.v_head_dim
        self.v_scale = config.value_scale
        self.window = config.sliding_window if sliding else None
        self.rope_theta = config.rope_for(layer_type).rope_theta
        self.rope_dims = config.rope_dims(layer_type)
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.v_head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.v_head_dim, hidden, bias=False)
        if sliding:
            self.sinks = mx.zeros((self.heads,))

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.rope_dims, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | str | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        if self.window is None:
            return "causal"
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q = self.q_proj(x).reshape(1, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(x).reshape(1, length, self.kv_heads, self.v_head_dim).transpose(0, 2, 1, 3)
            * self.v_scale
        )
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        mask = self.mask(length, keys.shape[2])
        sinks = self.sinks if "sinks" in self else None
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset), keys, values,
            scale=self.head_dim**-0.5,
            mask=mask,
            sinks=sinks,
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.v_head_dim)
        )
