import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache, SharedKVReader
from mlx_omnia.engine.core.masks import SLIDING, causal_mask
from mlx_omnia.engine.models.gemma4.config import Gemma4TextConfig
from mlx_omnia.engine.models.gemma4.layers.norm import RMSNormNoScale
from mlx_omnia.engine.models.gemma4.layers.rope import (
    cos_sin_tables,
    manual_rope,
    proportional_inv_freq,
)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.layer_type = config.attention_types[layer_idx]
        self.sliding = self.layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self.scale = 1.0

        if self.sliding:
            self.head_dim = config.head_dim
            rope = config.rope_parameters.sliding_attention
        else:
            self.head_dim = config.full_head_dim
            rope = config.rope_parameters.full_attention
        self.rope_type = rope.rope_type
        self.rope_theta = rope.rope_theta
        self.partial_rotary_factor = rope.partial_rotary_factor
        self.is_full = not self.sliding

        self.k_eq_v = config.attention_k_eq_v and not self.sliding

        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        if self.k_eq_v and config.num_global_key_value_heads is not None:
            self.kv_heads = config.num_global_key_value_heads
        key_values = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, queries, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        kv_shared = config.is_kv_shared_layer(layer_idx)
        self.kv_shared = kv_shared
        if not kv_shared:
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_proj = nn.Linear(hidden, key_values, bias=False)
            if not self.k_eq_v:
                self.v_proj = nn.Linear(hidden, key_values, bias=False)
            self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # Precompute proportional inv_freq for full layers (manual rope).
        if self.is_full and self.rope_type == "proportional":
            self._inv_freq = proportional_inv_freq(
                self.head_dim, self.partial_rotary_factor, self.rope_theta
            )
        else:
            self._inv_freq = None

    def mask(self, queries: int, keys: int) -> mx.array | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | SharedKVReader,
    ) -> mx.array:
        length = x.shape[1]

        if self.kv_shared:
            assert isinstance(cache, SharedKVReader)
            q = self.q_proj(x).reshape(
                1, length, self.heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            q = self.q_norm(q)
            q = self._apply_rope(q, cache.offset)
            keys = cache.keys
            values = cache.values
            assert keys is not None and values is not None
            attended = mx.fast.scaled_dot_product_attention(
                q,
                keys,
                values,
                scale=self.scale,
                mask=self.mask(length, keys.shape[2]),
            )
            return self.o_proj(
                attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
            )

        q = self.q_proj(x).reshape(
            1, length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(
            1, length, self.kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        if self.k_eq_v:
            v = k
        else:
            v = self.v_proj(x).reshape(
                1, length, self.kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q = self._apply_rope(q, cache.offset)
        k = self._apply_rope(k, cache.offset)

        assert isinstance(cache, KVCache)
        keys, values = cache.update_and_fetch(k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q,
            keys,
            values,
            scale=self.scale,
            mask=self.mask(length, keys.shape[2]),
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )

    def _rope_sliding(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

    def _rope_full(self, x: mx.array, offset: int, length: int) -> mx.array:
        assert self._inv_freq is not None
        positions = mx.arange(offset, offset + length, dtype=mx.int32)
        cos, sin = cos_sin_tables(self._inv_freq, positions, self.head_dim)
        return manual_rope(x, cos, sin)

    def _apply_rope(self, x: mx.array, offset: int) -> mx.array:
        length = x.shape[2]
        if self.is_full and self.rope_type == "proportional":
            return self._rope_full(x, offset, length)
        return self._rope_sliding(x, offset)
