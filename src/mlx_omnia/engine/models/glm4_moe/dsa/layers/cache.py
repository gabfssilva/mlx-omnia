from collections.abc import Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.cache import (
    BatchedKVCache,
    Composite,
    FixedKVCache,
    KVCache,
    LayerCache,
)

type DSASolo = DSACache | FixedDSACache
"""One sequence's layer state, growing or promoted."""

type DSAStore = DSASolo | BatchedDSACache
"""What one DSA layer reads: its own cache, or a row of a ragged batch of them."""


class DSACache(Composite):
    """One layer's two histories: the attention's keys/values and the indexer's keys.

    They advance together and rewind together, so the layer presents a single `offset`
    and a single `trim`; `generate.py` never sees that there are two.
    """

    def __init__(self) -> None:
        # The base sets `offset = 0` through the property below, which delegates to the
        # sub-caches — so they must exist first.
        self.attention = KVCache()
        self.index = KVCache()
        super().__init__()

    @property
    def parts(self) -> Mapping[str, LayerCache]:
        return {"attention": self.attention, "index": self.index}

    @property
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value

    @property
    def is_trimmable(self) -> bool:
        return self.attention.is_trimmable and self.index.is_trimmable

    def trim(self, length: int) -> None:
        self.attention.trim(length)
        self.index.trim(length)

    @property
    def is_fixable(self) -> bool:
        return self.attention.is_fixable and self.index.is_fixable

    def fixed(self, capacity: int) -> "FixedDSACache":
        """Both halves into fixed buffers of the same capacity.

        The two advance together and are read at the same columns — the indexer selects
        columns of the attention's history — so a capacity that differed between them
        would be a selection indexing rows the attention does not hold."""
        return FixedDSACache(
            FixedKVCache.promote(self.attention, capacity),
            FixedKVCache.promote(self.index, capacity),
        )

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedDSACache":
        """The adapter this layer is read through when the rows are ragged. Generic code
        cannot know this cache, so the dispatch is inverted and lands here."""
        caches: list[DSACache] = []
        for row in rows:
            if not isinstance(row, DSACache):
                raise TypeError(f"a batched DSA layer mixes {type(row).__name__} with DSACache")
            caches.append(row)
        return BatchedDSACache(caches)


class FixedDSACache(Composite):
    """The same two histories in the shape a traced decode can hold.

    Both halves are plain fixed buffers: the storage was never what made this family
    awkward to trace. What the promotion buys the layer is the position as a graph tensor
    — the rope offset of both the attention and the indexer — and `readable`'s band over
    the written columns, which the indexer masks its scores with *before* selecting and
    the attention ANDs the selection against afterwards.
    """

    def __init__(self, attention: FixedKVCache, index: FixedKVCache) -> None:
        self.attention = attention
        self.index = index

    @property
    def parts(self) -> Mapping[str, LayerCache]:
        return {"attention": self.attention, "index": self.index}

    @property
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value
        self.index.offset = value

    @property
    def position(self) -> mx.array:
        """Where both halves stand, off the graph. They are written by the same step from
        the same row, so one tensor answers for both — and it is the attention's, because
        that is the buffer whose columns the selection indexes."""
        return self.attention.position

    @property
    def span(self) -> int:
        return self.attention.span

    @property
    def rows(self) -> int:
        return self.attention.rows

    @property
    def is_fixable(self) -> bool:
        return True

    def fixed(self, capacity: int) -> "LayerCache":
        if self.attention.span == capacity:
            return self
        return FixedDSACache(
            _regrown(self.attention, capacity), _regrown(self.index, capacity)
        )

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, length: int) -> None:
        raise NotImplementedError("a compiled cache cannot be rewound")

    def rewind(self, kept: int, position: int) -> None:
        self.attention.rewind(kept, position)
        self.index.rewind(kept, position)


def _regrown(cache: FixedKVCache, capacity: int) -> FixedKVCache:
    grown = cache.fixed(capacity)
    assert isinstance(grown, FixedKVCache)
    return grown


class BatchedDSACache(LayerCache):
    """N `DSACache`s as one ragged layer.

    Selection is per row — each row's history has its own length, and its own answer to
    whether the indexer's top-k has anything to drop — so this adapter carries the rows
    themselves and the attention runs its body once per row. The two sub-views are the
    ordinary attention adapters over each half.
    """

    def __init__(self, caches: Sequence[DSACache]) -> None:
        self._caches = tuple(caches)
        self.attention = BatchedKVCache([cache.attention for cache in self._caches])
        self.index = BatchedKVCache([cache.index for cache in self._caches])

    @property
    def sequences(self) -> tuple[DSACache, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)
