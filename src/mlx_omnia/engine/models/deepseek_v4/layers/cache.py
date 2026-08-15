from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.batching import RaggedAdapter, RaggedBatchable
from mlx_omnia.engine.core.cache import (
    Composite,
    KVCache,
    LayerCache,
    Layout,
    Rows,
    Snapshot,
    reserve,
)
from mlx_omnia.engine.models.deepseek_v4.config import LOCAL, OVERLAP


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
        tail_kv, tail_gate = self.tail_kv, self.tail_gate
        if tail_kv is None or tail_gate is None:
            tail_kv = mx.zeros((1, self.ratio, kv.shape[-1]), dtype=kv.dtype)
            tail_gate = mx.zeros((1, self.ratio, gate.shape[-1]), dtype=gate.dtype)
            self.tail_kv, self.tail_gate = tail_kv, tail_gate

        total = length + self.remainder
        usable = total // self.ratio * self.ratio
        rest = total % self.ratio
        if usable:
            ready_kv = mx.concatenate(
                [tail_kv[:, : self.remainder], kv[:, : usable - self.remainder]], axis=1
            )
            ready_gate = mx.concatenate(
                [tail_gate[:, : self.remainder], gate[:, : usable - self.remainder]], axis=1
            )
            base = offset - self.remainder
            self.remainder = 0
        else:
            ready_kv, ready_gate, base = kv[:, :0], gate[:, :0], 0

        if rest:
            tail_kv[:, self.remainder : rest] = kv[:, -rest:]
            tail_gate[:, self.remainder : rest] = gate[:, -rest:]
        self.remainder = rest
        return ready_kv, ready_gate, base

    def append(self, pooled: mx.array) -> None:
        needed = self.pooled_rows + pooled.shape[2]
        buffer = reserve(self.pooled, needed, pooled)
        self.pooled = buffer
        buffer[..., self.pooled_rows : needed, :] = pooled
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

    @property
    def layout(self) -> Mapping[str, Layout]:
        """The pooled rows compose at one row per `ratio` tokens; the overlap carry does not.


        The tail is neither, and it is nowhere: on a boundary that is a multiple of `ratio` —
        which the span is required to be — `remainder` is zero and the tail buffer holds
        nothing a next call will read. `pooled_rows` is `offset // ratio` and is recomputed
        on the way in rather than carried, for the same reason `QuantizedKVCache` recomputes
        its split: a second copy of a rule is a second rule, free to disagree.
        """
        held: dict[str, Layout] = {"pooled": Rows(stride=self.ratio)}
        if self.ratio == OVERLAP:
            # Only the overlapping ratio carries anything across a window: each of its windows
            # pools its own lane B with the previous one's lane A, so the last raw window is
            # state. Every other ratio pools a window out of its own tokens and keeps nothing
            # — and declaring a carry it never produces would be a trunk that claims an anchor
            # and hands over none, which is silent zero reuse for the whole model.
            held["carry.kv"] = Snapshot()
            held["carry.gate"] = Snapshot()
        return held

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        if start % self.ratio or stop % self.ratio:
            raise ValueError(
                f"a pool of {self.ratio} cannot cut [{start}, {stop}): a span has to close "
                "the window, or its last row is built from tokens on both sides of the cut"
            )
        held: dict[str, mx.array] = {}
        if self.pooled is not None:
            held["pooled"] = self.pooled[..., start // self.ratio : stop // self.ratio, :]
        if self.previous is not None:
            held["carry.kv"], held["carry.gate"] = self.previous
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self.remainder = 0
        self.pooled = tensors.get("pooled")
        self.pooled_rows = offset // self.ratio
        kv, gate = tensors.get("carry.kv"), tensors.get("carry.gate")
        self.previous = None if kv is None or gate is None else (kv, gate)


class DeepseekV4Cache(Composite, RaggedBatchable):
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
    def parts(self) -> Mapping[str, LayerCache]:
        """The local window always, the two pools when this layer carries them. Absent and
        not empty: a layer with no compressor has no compressed rows to name, and a name
        that is sometimes empty is a name a resume would look for and not find."""
        held: dict[str, LayerCache] = {"attention": self.attention}
        if self.compressor is not None:
            held["compressor"] = self.compressor
        if self.indexer is not None:
            held["indexer"] = self.indexer
        return held

    @property
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value

    @property
    def is_trimmable(self) -> bool:
        return self.compressor is None and self.attention.is_trimmable

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


class BatchedDeepseekV4Cache(RaggedAdapter):
    """N `DeepseekV4Cache`s as one ragged layer: the rows, and where each one stands."""

    def __init__(self, caches: Sequence[DeepseekV4Cache]) -> None:
        self._caches = tuple(caches)

    @property
    def rows(self) -> tuple[DeepseekV4Cache, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)
