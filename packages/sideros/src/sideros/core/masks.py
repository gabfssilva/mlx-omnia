"""Attention masks and the layer-type names an interleaved schedule uses."""

import mlx.core as mx

SLIDING = "sliding_attention"
FULL = "full_attention"


def causal_mask(queries: int, keys: int, window: int | None) -> mx.array:
    """Which keys each query may attend to: everything up to itself, and within `window`
    positions of it when the layer slides. The queries are the *last* `queries` of the
    `keys` positions, which is what lets a single query attend to a whole cache."""
    rows = mx.arange(keys - queries, keys).reshape(queries, 1)
    columns = mx.arange(keys).reshape(1, keys)
    within = rows >= columns
    if window is None:
        return within
    return within & (rows < columns + window)
