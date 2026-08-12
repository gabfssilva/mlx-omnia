import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.qwen3_next.config import Qwen3NextConfig


class Qwen3NextAttention(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dims = config.rope_dims
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        bias = config.attention_bias
        # Twice as wide: the second half is the output gate, not a query.
        self.q_proj = nn.Linear(hidden, 2 * self.heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.rope_dims, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        width = self.heads * self.head_dim
        q, gate = mx.split(self.q_proj(x).reshape(1, length, self.heads, -1), 2, axis=-1)
        k = self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim)
        keys, values = cache.update_and_fetch(
            self.rope(self.k_norm(k).transpose(0, 2, 1, 3), offset), v.transpose(0, 2, 1, 3)
        )
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset), keys, values,
            scale=self.head_dim**-0.5,
            mask=None if length == 1 else "causal",
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, width)
        return self.o_proj(output * mx.sigmoid(gate.reshape(1, length, width)))
