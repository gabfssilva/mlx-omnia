import math

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import AttentionMask, attend
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.core.kernels.attention import (
    AttentionStep,
    AttentionStepStrategy,
    rotate,
)
from mlx_omnia.engine.models.gpt_oss.config import GPTOSSConfig
from mlx_omnia.engine.models.gpt_oss.layers import flags


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

    def __call__(self, x: mx.array, mask: AttentionMask, cache: LayerCache) -> mx.array:
        batch, length = x.shape[0], x.shape[1]
        query_width = self.heads * self.head_dim
        key_value_width = self.kv_heads * self.head_dim
        q, k, v = mx.split(
            self.qkv_proj(x), [query_width, query_width + key_value_width], axis=-1
        )
        if isinstance(cache, KVCache) and batch == 1:
            attended = self._attention(cache, x.dtype)(q, k, v, mask)
        else:
            attended = self._attend(q, k, v, mask, cache, batch, length)
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, query_width))

    def _attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: AttentionMask,
        cache: LayerCache,
        batch: int,
        length: int,
    ) -> mx.array:
        """The general path: the rotation the fused step does inside, done in ops here, and
        the cache read through the door. Same freqs, same mscale, same partial geometry."""
        offset = cache.offset
        queries = self._heads(q, self.heads, batch, length)
        keys = self._heads(k, self.kv_heads, batch, length)
        values = self._heads(v, self.kv_heads, batch, length)
        queries = self._rotate(queries, offset)
        keys = self._rotate(keys, offset)
        return attend(
            cache,
            queries,
            keys=keys,
            values=values,
            scale=self.scale,
            mask=mask,
            sinks=self.sinks,
        )

    def _heads(self, raw: mx.array, count: int, batch: int, length: int) -> mx.array:
        return raw.reshape(batch, length, count, self.head_dim).transpose(0, 2, 1, 3)

    def _rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return rotate(
            x,
            rotary_pairs=self.head_dim // 2,
            freqs=self._freqs,
            base=0.0,
            mscale=self._mscale,
            offset=offset,
        )

    def _attention(self, cache: KVCache, dtype: mx.Dtype) -> AttentionStepStrategy:
        """Resolved once, at the first step after load — and again whenever the cache is
        replaced or the bench switch moves, since the step is bound to one cache."""
        step = self._step
        fused = flags.USE_SINK_ATTENTION
        if step is None or self._step_cache is not cache or self._step_sink != fused:
            step = AttentionStep(
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
                reference=not fused,
            )
            self._step, self._step_cache, self._step_sink = step, cache, fused
        return step
