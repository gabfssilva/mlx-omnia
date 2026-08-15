from collections.abc import Callable, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.cache import KVCache, LayerCache, reserve
from mlx_omnia.engine.models.deepseek_v4.config import LOCAL


class PoolCache(LayerCache):
    """One compressor's history: the pooled rows, plus the tail of tokens that has not
    completed a window yet.

    The pooled rows grow into a block-reserved buffer like `KVCache`'s (one row per
    `ratio` tokens, so the growth is rare). Nothing here rewinds: the tail buffer and the
    overlap carry are consumed as they are written, and a trimmed window cannot be pooled
    again from rows the cache no longer holds — the same reason DeltaNet's state closes
    speculation.
    """

    def __init__(self, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.tail_kv: mx.array | None = None
        self.tail_gate: mx.array | None = None
        self.remainder = 0
        self.pooled: mx.array | None = None
        # Not `rows`: the base answers that name with the offset, and this counter is the
        # pooled rows written — one per `ratio` tokens, not one per token.
        self.pooled_rows = 0
        self.previous: tuple[mx.array, mx.array] | None = None

    @property
    def is_replayable(self) -> bool:
        """No. `checkpoint()` cannot capture the tail: a call that completes a window reads
        `tail[:remainder]` and then overwrites those same rows with what is left over, so a
        restore rewinds the counter onto rows the round already destroyed. A replay would
        pool a window out of the wrong tokens and say nothing."""
        return False

    @property
    def nbytes(self) -> int:
        buffers = (self.tail_kv, self.tail_gate, self.pooled)
        return sum(buffer.nbytes for buffer in buffers if buffer is not None)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        buffers = (self.tail_kv, self.tail_gate, self.pooled, *(self.previous or ()))
        return tuple(buffer for buffer in buffers if buffer is not None)

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        state = (self.remainder, self.pooled_rows, self.previous, self.pooled)

        def restore() -> None:
            parent()
            self.remainder, self.pooled_rows, self.previous, self.pooled = state

        return restore

    def accumulate(
        self, kv: mx.array, gate: mx.array, offset: int
    ) -> tuple[mx.array, mx.array, int]:
        """The rows that complete whole windows, and the absolute position of the first of
        them. What is left over is kept for the next call."""
        length = kv.shape[1]
        if self.tail_kv is None or self.tail_gate is None:
            self.tail_kv = mx.zeros((1, self.ratio, kv.shape[-1]), dtype=kv.dtype)
            self.tail_gate = mx.zeros((1, self.ratio, gate.shape[-1]), dtype=gate.dtype)

        total = length + self.remainder
        usable = total // self.ratio * self.ratio
        rest = total % self.ratio
        if usable:
            ready_kv = mx.concatenate(
                [self.tail_kv[:, : self.remainder], kv[:, : usable - self.remainder]], axis=1
            )
            ready_gate = mx.concatenate(
                [self.tail_gate[:, : self.remainder], gate[:, : usable - self.remainder]], axis=1
            )
            base = offset - self.remainder
            self.remainder = 0
        else:
            ready_kv, ready_gate, base = kv[:, :0], gate[:, :0], 0

        if rest:
            self.tail_kv[:, self.remainder : rest] = kv[:, -rest:]
            self.tail_gate[:, self.remainder : rest] = gate[:, -rest:]
        self.remainder = rest
        return ready_kv, ready_gate, base

    def append(self, pooled: mx.array) -> None:
        needed = self.pooled_rows + pooled.shape[2]
        self.pooled = reserve(self.pooled, needed, pooled)
        self.pooled[..., self.pooled_rows : needed, :] = pooled
        self.pooled_rows = needed

    def fetch(self, head_dim: int, dtype: mx.Dtype) -> mx.array:
        if self.pooled is None:
            return mx.zeros((1, 1, 0, head_dim), dtype=dtype)
        return self.pooled[..., : self.pooled_rows, :]

    def mask(self, length: int, offset: int) -> mx.array | None:
        """Pooled row `i` is visible to the query at absolute position `p` while
        `i < (p + 1) // ratio`. At T=1 every row already in the cache passes."""
        if length == 1:
            return None
        rows = mx.arange(offset + 1, offset + length + 1).reshape(-1, 1)
        return mx.arange(self.pooled_rows).reshape(1, -1) < rows // self.ratio


class DeepseekV4Cache(LayerCache):
    """One layer's histories: the local keys (K == V, one buffer), the compressor's pooled
    rows and — on the indexed layers — the indexer's own pooled rows. They advance
    together and the layer presents a single `offset`."""

    def __init__(self, ratio: int, indexed: bool) -> None:
        # Before `super().__init__()`: the base sets `offset`, and this class answers that
        # name with the local cache's.
        self.attention = KVCache()
        self.compressor = PoolCache(ratio) if ratio != LOCAL else None
        self.indexer = PoolCache(ratio) if indexed else None
        super().__init__()

    @property
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value

    @property
    def is_trimmable(self) -> bool:
        return self.compressor is None and self.attention.is_trimmable

    @property
    def is_replayable(self) -> bool:
        """The pools answer for themselves — a layer that carries one cannot be rewound by
        either road, and saying otherwise is what lets speculation run off a pooled window
        that was built from tokens the sequence never had."""
        pools = (self.compressor, self.indexer)
        return self.attention.is_replayable and all(
            pool.is_replayable for pool in pools if pool is not None
        )

    @property
    def nbytes(self) -> int:
        pools = (self.compressor, self.indexer)
        return self.attention.nbytes + sum(pool.nbytes for pool in pools if pool is not None)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        pools = (self.compressor, self.indexer)
        held = self.attention.tensors
        for pool in pools:
            if pool is not None:
                held += pool.tensors
        return held

    def checkpoint(self) -> Callable[[], None]:
        pools = (self.compressor, self.indexer)
        restores = [self.attention.checkpoint()]
        restores.extend(pool.checkpoint() for pool in pools if pool is not None)

        def restore() -> None:
            for undo in restores:
                undo()

        return restore

    def trim(self, length: int) -> None:
        if self.compressor is not None:
            raise NotImplementedError("a pooled window cannot be recompressed after a trim")
        self.attention.trim(length)

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedDeepseekV4Cache":
        """This layer's rows as one ragged batch. The pools' state is scalar *per row* —
        a partial tail, a remainder, a pooled length that differs between rows — so the
        adapter holds the rows and the forward walks them, rather than stacking anything."""
        held: list[DeepseekV4Cache] = []
        for row in rows:
            if not isinstance(row, DeepseekV4Cache):
                raise TypeError(f"a batched deepseek_v4 layer mixes in {type(row).__name__}")
            if (row.compressor is None) != (self.compressor is None) or (
                row.indexer is None
            ) != (self.indexer is None):
                raise TypeError("a batched deepseek_v4 layer mixes compress ratios")
            held.append(row)
        return BatchedDeepseekV4Cache(held)


class BatchedDeepseekV4Cache:
    """N `DeepseekV4Cache`s as one ragged layer: the rows, and where each one stands."""

    def __init__(self, caches: Sequence[DeepseekV4Cache]) -> None:
        self._caches = tuple(caches)

    @property
    def rows(self) -> tuple[DeepseekV4Cache, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)
