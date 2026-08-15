from collections.abc import Callable, Sequence

import mlx.core as mx

from mlx_omnia.engine.batching import BatchedKVCache, RaggedAdapter, RaggedBatchable
from mlx_omnia.engine.core.cache import KVCache, LayerCache

type DSAStore = DSACache | BatchedDSACache
"""What one DSA layer reads: its own cache, or a row of a ragged batch of them."""


class DSACache(LayerCache, RaggedBatchable):
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
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value

    @property
    def is_trimmable(self) -> bool:
        return self.attention.is_trimmable and self.index.is_trimmable

    @property
    def nbytes(self) -> int:
        return self.attention.nbytes + self.index.nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return self.attention.tensors + self.index.tensors

    def checkpoint(self) -> Callable[[], None]:
        restores = (self.attention.checkpoint(), self.index.checkpoint())

        def restore() -> None:
            for undo in restores:
                undo()

        return restore

    def trim(self, length: int) -> None:
        self.attention.trim(length)
        self.index.trim(length)

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedDSACache":
        """The adapter this layer is read through when the rows are ragged. Generic code
        cannot know this cache, so the dispatch is inverted and lands here."""
        caches: list[DSACache] = []
        for row in rows:
            if not isinstance(row, DSACache):
                raise TypeError(f"a batched DSA layer mixes {type(row).__name__} with DSACache")
            caches.append(row)
        return BatchedDSACache(caches)


class BatchedDSACache(RaggedAdapter):
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
    def rows(self) -> tuple[DSACache, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)
