import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.nemotron_h.config import NemotronHConfig


class NemotronHAttention(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=config.attention_bias)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        heads = [
            projection(x).reshape(1, length, count, self.head_dim).transpose(0, 2, 1, 3)
            for projection, count in (
                (self.q_proj, self.heads),
                (self.k_proj, self.kv_heads),
                (self.v_proj, self.kv_heads),
            )
        ]
        keys, values = cache.update_and_fetch(heads[1], heads[2])
        attended = mx.fast.scaled_dot_product_attention(
            heads[0], keys, values,
            scale=self.head_dim**-0.5,
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
