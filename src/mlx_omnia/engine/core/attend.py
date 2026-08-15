"""The door every attention goes through to reach its cache.

Today a layer calls `cache.update_and_fetch(k, v)` and hands the two dense tensors to
`mx.fast.scaled_dot_product_attention`. A cache that stores its rows compressed cannot
answer that call: returning dense K and V would materialize exactly the bytes the
compression avoided. So the read becomes an operation *of the cache* rather than something
done to it, and this is the one function a layer calls.

A cache that does not implement `Attending` — every one in `core.cache` — takes the dense
path here, in the op order it had before this function existed. That is not a tolerance: it
is the same two calls with the same arguments, so a family that adheres is bit-identical to
one that has not.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx

from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, RingKVCache
from mlx_omnia.engine.core.mxcompat import softmax

__all__ = ["Attending", "AttentionMask", "KVStore", "attend", "softcapped_attention"]

type AttentionMask = mx.array | str | None

type Storing = KVCache | FixedKVCache | RingKVCache
"""The caches that answer `update_and_fetch`. Named here rather than imported from
`core.attention` because that module reads this one."""


@runtime_checkable
class Attending(Protocol):
    """A cache that reads itself. What it must not do is hand back rows.

    `keys` and `values` are the ones this step produced, not the history: writing them is
    part of attending, because a compressed cache decides for itself which of its regions
    the new rows land in.
    """

    offset: int | mx.array

    def attend(
        self,
        queries: mx.array,
        *,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: AttentionMask,
        sinks: mx.array | None = None,
        softcap: float | None = None,
    ) -> mx.array: ...


type KVStore = Storing | Attending


def softcapped_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    cap: float,
    mask: mx.array | None,
) -> mx.array:
    """Attention with the logits softcapped — `tanh(s/cap) * cap` between QK and softmax.

    `mx.fast.scaled_dot_product_attention` has no cap parameter, so this is the manual
    softmax in the op order the capped families use: queries scaled *before* the matmul,
    GQA grouped by expanding the KV head axis, boolean mask applied as a `finfo.min` fill,
    softmax accumulated in fp32. Floating-point order is part of the behavior — do not
    reassociate the scale or the grouping.
    """
    batch, heads, length, head_dim = queries.shape
    kv_heads = keys.shape[1]
    repeats = heads // kv_heads
    queries = queries * scale
    if repeats > 1:
        queries = queries.reshape(batch, kv_heads, repeats, length, head_dim)
        keys = mx.expand_dims(keys, 2)
        values = mx.expand_dims(values, 2)
    scores = mx.tanh((queries @ keys.swapaxes(-1, -2)) / cap) * cap
    if mask is not None:
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    attended = softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype) @ values
    if repeats > 1:
        attended = attended.reshape(batch, heads, length, head_dim)
    return attended


def attend(
    cache: KVStore | None,
    queries: mx.array,
    *,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: AttentionMask,
    sinks: mx.array | None = None,
    softcap: float | None = None,
) -> mx.array:
    """The step's rows into the cache, and the attention over everything it holds.

    `cache is None` is the cacheless forward — a prefill nobody is going to continue, which
    is how the parity fixtures are generated — and it attends the rows it was handed.

    `sinks` rides through because it changes the softmax and not the rows: a cache that
    cannot honour it must refuse rather than attend without it — attending sinkless is a
    different model. `softcap` is the same kind of parameter, and it forces the manual
    softmax because the fused kernel has no cap.
    """
    if isinstance(cache, Attending):
        return cache.attend(
            queries, keys=keys, values=values, scale=scale, mask=mask, sinks=sinks, softcap=softcap
        )
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    if softcap is not None:
        if isinstance(mask, str):
            raise ValueError("a softcapped attention needs an explicit boolean mask, not " + mask)
        if sinks is not None:
            raise ValueError("softcap and sinks do not compose; no family asks for both")
        return softcapped_attention(queries, keys, values, scale=scale, cap=softcap, mask=mask)
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=sinks
    )
