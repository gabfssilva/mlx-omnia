import math

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.core.layers import split_qkv
from mlx_omnia.models.llama4.config import Llama4TextConfig

# The reference implementation hardcodes 1e-6; transformers uses rms_norm_eps (1e-5).
# The reference is bf16-vs-bf16, so we match it. The 10x difference is sub-ulp on normalized
# q/k, well below the bf16 floor.
QK_NORM_EPS = 1e-6


class Llama4Attention(nn.Module):
    """Per-layer RoPE/NoPE + qk_norm + temperature tuning.

    RoPE (chunked) layers: rope → qk_norm (weightless). NoPE (full) layers:
    temperature-scale q, no qk_norm. `mask=None` at T=1, chunked/causal at prefill.
    """

    def __init__(self, config: Llama4TextConfig, layer_idx: int, freqs: mx.array) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        self.use_rope = bool(config.rope_flags[layer_idx])
        self.use_qk_norm = config.use_qk_norm and self.use_rope
        self.attn_temperature_tuning = bool(config.attn_temperature_tuning)
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self._freqs = freqs

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        if self.use_rope:
            q = self._rope(q, offset)
            k = self._rope(k, offset)
            if self.use_qk_norm:
                q = mx.fast.rms_norm(q, weight=None, eps=QK_NORM_EPS)
                k = mx.fast.rms_norm(k, weight=None, eps=QK_NORM_EPS)
        elif self.attn_temperature_tuning:
            positions = mx.arange(offset + 1, offset + length + 1, dtype=mx.float32)
            attn_scales = (
                mx.log(mx.floor(positions / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            q = (q * attn_scales[:, None]).astype(q.dtype)
        keys, values = cache.update_and_fetch(k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))
