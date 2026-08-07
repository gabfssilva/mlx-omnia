import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.sink_attention import sink_attention, sink_attention_applies
from sideros.core.layers import split_qkv
from sideros.models.gpt_oss.config import GPTOSSConfig
from sideros.models.gpt_oss.layers import flags


class GPTOSSAttention(nn.Module):
    """GQA with bias on every projection, no q/k norm, and a per-head sink logit that
    sits in the softmax denominator. q‖k‖v is one leaf, concatenated at load."""

    def __init__(self, config: GPTOSSConfig, freqs: mx.array, mscale: float) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, queries + 2 * key_values, bias=True)
        self.o_proj = nn.Linear(queries, config.hidden_size, bias=True)
        self.sinks = mx.zeros((self.heads,))
        # Leading underscore: not a parameter, so the strict load does not demand it.
        self._freqs = freqs
        self._mscale = mscale

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
        queries = self._rope(q, offset)
        keys, values = cache.update_and_fetch(self._rope(k, offset), v)
        sinks = self.sinks
        if (
            flags.USE_SINK_ATTENTION
            and not isinstance(mask, str)
            and sinks.dtype == queries.dtype
            and sink_attention_applies(queries, keys)
        ):
            attended = sink_attention(queries, keys, values, sinks, mask, self.scale)
        else:
            attended = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask, sinks=sinks
            )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        scaled = x * self._mscale if self._mscale != 1.0 else x
        return mx.fast.rope(
            scaled, self.head_dim, traditional=False, base=None, scale=1.0, offset=offset,
            freqs=self._freqs,
        )
