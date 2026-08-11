import math

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.core.kernels.attention import (
    AttentionStep,
    AttentionStepStrategy,
    DefaultAttentionStep,
)
from mlx_omnia.models.gpt_oss.config import GPTOSSConfig
from mlx_omnia.models.gpt_oss.layers import flags


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
        self._step: AttentionStepStrategy | None = None
        self._step_cache: KVCache | None = None
        self._step_sink = flags.USE_SINK_ATTENTION

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        length = x.shape[1]
        query_width = self.heads * self.head_dim
        key_value_width = self.kv_heads * self.head_dim
        q, k, v = mx.split(
            self.qkv_proj(x), [query_width, query_width + key_value_width], axis=-1
        )
        attended = self._attention(cache, x.dtype)(q, k, v, mask)
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))

    def _attention(self, cache: KVCache, dtype: mx.Dtype) -> AttentionStepStrategy:
        """Resolved once, at the first step after load — and again whenever the cache is
        replaced or the bench switch moves, since the step is bound to one cache."""
        step = self._step
        fused = flags.USE_SINK_ATTENTION
        if step is None or self._step_cache is not cache or self._step_sink != fused:
            step = (
                AttentionStep(
                    cache,
                    heads=self.heads,
                    kv_heads=self.kv_heads,
                    head_dim=self.head_dim,
                    scale=self.scale,
                    dtype=dtype,
                    rotary_pairs=self.head_dim // 2,
                    mscale=self._mscale,
                    freqs=self._freqs,
                    sinks=self.sinks,
                )
                if fused
                else DefaultAttentionStep(
                    cache=cache,
                    heads=self.heads,
                    kv_heads=self.kv_heads,
                    head_dim=self.head_dim,
                    scale=self.scale,
                    query_weight=None,
                    key_weight=None,
                    eps=1e-6,
                    rotary_pairs=self.head_dim // 2,
                    mscale=self._mscale,
                    freqs=self._freqs,
                    base=0.0,
                    sinks=self.sinks,
                )
            )
            self._step, self._step_cache, self._step_sink = step, cache, fused
        return step
