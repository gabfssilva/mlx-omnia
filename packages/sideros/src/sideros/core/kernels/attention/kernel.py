"""The attention step's contract: what a strategy is and what a model declares.

The primitive is the whole T=1 attention step, taken at its widest fused form: the raw
q/k/v projections in, the context out, with the Q/K norm, the rotation and the cache
write inside. A declaration therefore carries everything those three stages need — the
norm gains and their epsilon, the rotary geometry (an angle row per position for the
fused kernels, the frequency table for the ops chain), the cache the step writes into,
and the sink logits when the layer has them. The window is the ring cache's own.

`(raw_queries [1, T, heads*head_dim], raw_keys/raw_values [1, T, kv_heads*head_dim],
mask) -> context [1, heads, T, head_dim]` — the shape
`mx.fast.scaled_dot_product_attention` returns. The step reads the write position off
the cache and advances it; the caller does not.
"""

from collections.abc import Callable
from typing import Protocol

import mlx.core as mx

from sideros.core.cache import FixedKVCache, KVCache, RingKVCache

AttentionCache = KVCache | FixedKVCache | RingKVCache

Angles = Callable[[int | mx.array], mx.array]
"""A position's rotation as `[cos(0..r-1), sin(0..r-1)]` float32, `r` the pair count.

The fused kernels take the row, not the table: deriving it per step costs more ops than
the rotation it feeds, so the caller owns an atlas and this is a lookup into it.
"""


class AttentionStepStrategy(Protocol):
    """(raw_queries, raw_keys, raw_values, mask) -> context [1, heads, T, head_dim]."""

    def __call__(
        self,
        raw_queries: mx.array,
        raw_keys: mx.array,
        raw_values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array: ...
