import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.masks import causal_mask
from sideros.models.afmoe.config import AfmoeConfig


class AfmoeAttention(nn.Module):
    def __init__(self, config: AfmoeConfig, sliding: bool) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rotary = sliding
        self.window = config.sliding_window if sliding else None
        hidden = config.hidden_size
        self.queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, self.queries, bias=False)
        self.k_proj = nn.Linear(hidden, key_values, bias=False)
        self.v_proj = nn.Linear(hidden, key_values, bias=False)
        self.o_proj = nn.Linear(self.queries, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, self.queries, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        if not self.rotary:
            return x
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
        heads = [
            projection(x).reshape(1, length, count, self.head_dim).transpose(0, 2, 1, 3)
            for projection, count in (
                (self.q_proj, self.heads),
                (self.k_proj, self.kv_heads),
                (self.v_proj, self.kv_heads),
            )
        ]
        keys, values = cache.update_and_fetch(self.rope(self.k_norm(heads[1]), offset), heads[2])
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(self.q_norm(heads[0]), offset), keys, values,
            scale=self.head_dim**-0.5,
            mask=self.mask(length, keys.shape[2]),
        )
        gated = attended.transpose(0, 2, 1, 3).reshape(1, length, self.queries) * mx.sigmoid(
            self.gate_proj(x)
        )
        return self.o_proj(gated)
