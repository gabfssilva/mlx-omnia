"""The universal attention step: the op chain every model already falls back to.

`build` accepts every declaration and every cache, so it registers last and makes the
delegator total. The step is the stock composition — `mx.fast.rms_norm` on q and k,
`mx.fast.rope` (the frequency table when the declaration carries one, the plain base
otherwise), `cache.update_and_fetch`, then `mx.fast.scaled_dot_product_attention` with
the sink logits when the layer has them. Six dispatches where the fused strategies spend
one, and the q/k/v round-trip through device memory between each; what defines it is
that no shape, dtype or cache layout can turn it away.

`mx.fast.scaled_dot_product_attention` is called by name rather than captured, so a
patched op — `attention.sdpa.SCALED_DOT_PRODUCT_ATTENTION` — still stands in for it.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.core.cache import FixedKVCache, RingKVCache
from mlx_omnia.core.kernels.attention.kernel import Angles, AttentionCache


def split_heads(raw: mx.array, count: int, head_dim: int) -> mx.array:
    """`[1, T, count*head_dim]` -> `[1, count, T, head_dim]`, the layout rope and SDPA
    both want."""
    return raw.reshape(1, raw.shape[1], count, head_dim).transpose(0, 2, 1, 3)


def qk_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    return x if weight is None else mx.fast.rms_norm(x, weight, eps)


def rotate(
    x: mx.array,
    *,
    rotary_pairs: int,
    freqs: mx.array | None,
    base: float,
    mscale: float,
    offset: int | mx.array,
) -> mx.array:
    """Partial RoPE with the YaRN length factor folded into the rotated block only —
    the rotated head prefix is pre-scaled, the passthrough tail is left alone."""
    if rotary_pairs == 0:
        return x
    rotary_dim = 2 * rotary_pairs
    if mscale != 1.0:
        x = mx.concatenate([x[..., :rotary_dim] * mscale, x[..., rotary_dim:]], axis=-1)
    return mx.fast.rope(
        x,
        rotary_dim,
        traditional=False,
        base=None if freqs is not None else base,
        scale=1.0,
        offset=offset,
        freqs=freqs,
    )


def write_position(cache: AttentionCache) -> int | mx.array:
    """The position this step writes at: the state tensor for a cache that keeps one, the
    host-side counter otherwise."""
    if isinstance(cache, (FixedKVCache, RingKVCache)):
        return cache.position
    return cache.offset


@dataclass(frozen=True)
class DefaultAttentionStep:
    cache: AttentionCache
    heads: int
    kv_heads: int
    head_dim: int
    scale: float
    query_weight: mx.array | None
    key_weight: mx.array | None
    eps: float
    rotary_pairs: int
    mscale: float
    freqs: mx.array | None
    base: float
    sinks: mx.array | None

    @classmethod
    def build(
        cls,
        cache: AttentionCache,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        scale: float,
        dtype: mx.Dtype,
        query_weight: mx.array | None,
        key_weight: mx.array | None,
        eps: float,
        rotary_pairs: int,
        mscale: float,
        angles: Angles | None,
        freqs: mx.array | None,
        base: float,
        sinks: mx.array | None,
    ) -> Self:
        return cls(
            cache,
            heads,
            kv_heads,
            head_dim,
            scale,
            query_weight,
            key_weight,
            eps,
            rotary_pairs,
            mscale,
            freqs,
            base,
            sinks,
        )

    def __call__(
        self,
        raw_queries: mx.array,
        raw_keys: mx.array,
        raw_values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        offset = write_position(self.cache)
        queries = split_heads(raw_queries, self.heads, self.head_dim)
        keys = split_heads(raw_keys, self.kv_heads, self.head_dim)
        values = split_heads(raw_values, self.kv_heads, self.head_dim)
        queries = self._rotate(qk_norm(queries, self.query_weight, self.eps), offset)
        keys = self._rotate(qk_norm(keys, self.key_weight, self.eps), offset)
        cached_keys, cached_values = self.cache.update_and_fetch(keys, values)
        return mx.fast.scaled_dot_product_attention(
            queries,
            cached_keys,
            cached_values,
            scale=self.scale,
            mask=mask,
            sinks=self.sinks,
        )

    def _rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return rotate(
            x,
            rotary_pairs=self.rotary_pairs,
            freqs=self.freqs,
            base=self.base,
            mscale=self.mscale,
            offset=offset,
        )
