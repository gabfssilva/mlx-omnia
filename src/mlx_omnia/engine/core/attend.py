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

    def attend(
        self,
        queries: mx.array,
        *,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: AttentionMask,
    ) -> mx.array: ...


def attend(
    cache: Storing | Attending | None,
    queries: mx.array,
    *,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: AttentionMask,
) -> mx.array:
    """The step's rows into the cache, and the attention over everything it holds.

    `cache is None` is the cacheless forward — a prefill nobody is going to continue, which
    is how the parity fixtures are generated — and it attends the rows it was handed.
    """
    if isinstance(cache, Attending):
        return cache.attend(queries, keys=keys, values=values, scale=scale, mask=mask)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)
