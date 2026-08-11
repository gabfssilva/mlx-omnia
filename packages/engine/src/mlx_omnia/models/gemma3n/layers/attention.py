import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache, SharedKVReader
from mlx_omnia.core.masks import SLIDING, causal_mask
from mlx_omnia.models.gemma3n.config import Gemma3nTextConfig


class Gemma3nAttention(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        sliding = config.layer_types[layer] == SLIDING
        self.window = config.sliding_window if sliding else None
        self.rope_base = config.rope_local_base_freq if sliding else config.rope_theta
        self.shared = layer >= config.first_shared_layer
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.eps = config.rms_norm_eps

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: KVCache | SharedKVReader) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q = self.q_norm(
            self.q_proj(x).reshape(1, length, self.heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        if self.shared:
            assert isinstance(cache, SharedKVReader)
            assert cache.keys is not None and cache.values is not None
            keys, values = cache.keys, cache.values
        else:
            assert isinstance(cache, KVCache)
            k = self.k_norm(
                self.k_proj(x).reshape(1, length, self.kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            # v_norm has no learned scale.
            v = mx.fast.rms_norm(
                self.v_proj(x).reshape(1, length, self.kv_heads, self.head_dim), None, self.eps
            ).transpose(0, 2, 1, 3)
            keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset), keys, values,
            scale=1.0,
            mask=self.mask(length, keys.shape[2]),
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )
