"""A layer that reads another layer's keys and values, and holds none of its own."""

from collections.abc import Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.attend import Attending, AttentionMask, dense_attention
from mlx_omnia.engine.core.cache.kv import BatchedKVCache, FixedKVCache, KVCache
from mlx_omnia.engine.core.cache.layer import LayerCache, row_mask, uniform
from mlx_omnia.engine.core.cache.layout import Layout

type KVWriter = KVCache | FixedKVCache
"""The layer a reader borrows from: growing, or promoted for a compiled decode."""


class SharedKVReader(LayerCache):
    """A cache placeholder for KV-shared layers: no projection, no own buffer.

    It reads the full-length K,V of the storing layer of the same type. The link is set
    per forward by the trunk, which is the only thing that knows which layer stores for
    which — and it is a **link and not a copy**. Copying the two tensors in would be right
    exactly once: `mx.compile` runs a function's Python a single time, so a reader holding
    the arrays the trace happened to see would attend those rows forever while the writer
    moved on. Holding the writer instead, `fetch` reads its current tensors at the moment
    the graph reads them — and the writer's layer runs first, so they are this step's.
    """

    offset: int

    def __init__(self) -> None:
        super().__init__()
        self.source: KVWriter | None = None
        self._step = 0

    def adopt(self, writer: KVWriter, length: int) -> None:
        """Take the layer that stores for this one, and how many rows this step wrote.

        `offset` is the host's view and lands where this layer's queries sit — before the
        step, because the writer already took it. `position` is the same number off the
        graph, which is what a trace has to rotate by.
        """
        self.source = writer
        self._step = length
        # A growing writer moved its own `offset` when it took these rows, so the step's
        # start is one behind it. A promoted one moved a graph position instead and left
        # `offset` where the step began, which is already the number wanted.
        self.offset = (
            writer.offset if isinstance(writer, FixedKVCache) else writer.offset - length
        )

    @property
    def position(self) -> int | mx.array:
        """Where this layer's queries sit, off the graph when the graph owns it."""
        writer = self._writer()
        if isinstance(writer, FixedKVCache):
            return writer.position - self._step
        return writer.offset - self._step

    @property
    def span(self) -> int:
        return self._writer().span

    def fetch(self) -> tuple[mx.array, mx.array]:
        """The writer's rows, read now rather than when the link was made."""
        return self._writer().fetch()

    def readable(self, mask: AttentionMask, queries: int, /) -> AttentionMask:
        """`LayerCache.readable`, delegated: the columns that exist are the writer's, and so
        is the question of whether its fetch hands back more of them than it wrote."""
        return self._writer().readable(mask, queries)

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)

    @property
    def is_fixable(self) -> bool:
        """Yes, and there is nothing to do about it: this layer holds no tensor of its own,
        so its fixed form is itself. What had to change first was the link — while the rows
        were copied into a plain attribute every forward, promoting the layer would have
        left a compiled decode attending whatever the trace happened to see."""
        return True

    def fixed(self, capacity: int) -> "LayerCache":
        del capacity
        return self

    def batched(self, rows: Sequence[LayerCache]) -> "LayerCache":
        return BatchedSharedKVReader(uniform(rows, SharedKVReader))

    @property
    def layout(self) -> Mapping[str, Layout]:
        """Nothing of its own. What it reads is the storing layer's, republished on every
        forward before the block runs, so a resumed trunk gets the link back from the store
        — and no span carries the same rows twice."""
        return {}

    def _writer(self) -> KVWriter:
        writer = self.source
        assert writer is not None, "a shared reader was read before its writer published"
        return writer


class BatchedSharedKVReader(LayerCache, Attending):
    """N `SharedKVReader`s as one attending layer.

    Ragged rows have no single dense tensor to attend, so the read becomes per-row: `adopt`
    walks the writer's rows and links each reader to its own, and `attend` then reads what
    each one links to. The step's `keys` and `values` are ignored because a reader writes
    nothing of its own.
    """

    def __init__(self, caches: Sequence[SharedKVReader]) -> None:
        self._caches = tuple(caches)

    @property
    def sequences(self) -> tuple[SharedKVReader, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        """Per-row positions, read where each row's writer keeps them: a promoted writer's
        live in the graph and its host offset is the value the trace was built with."""
        positions: list[mx.array] = []
        for cache in self._caches:
            at = cache.position
            positions.append(at if isinstance(at, mx.array) else mx.array([at], dtype=mx.int32))
        return mx.concatenate(positions)

    @property
    def span(self) -> int:
        """The widest row's. A reader holds no rows of its own, so what a mask over it must
        cover is what its writer holds."""
        return max(cache.span for cache in self._caches)

    def adopt(self, store: BatchedKVCache, length: int) -> None:
        for reader, writer in zip(self._caches, store.sequences, strict=True):
            if not isinstance(writer, KVCache | FixedKVCache):
                raise TypeError(f"a shared reader adopts dense rows; got {type(writer).__name__}")
            reader.adopt(writer, length)

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
    ) -> mx.array:
        attended: list[mx.array] = []
        for index, cache in enumerate(self._caches):
            history_keys, history_values = cache.fetch()
            effective = cache.readable(
                row_mask(mask, index, len(self._caches), history_keys.shape[2]),
                queries.shape[2],
            )
            attended.append(
                dense_attention(
                    queries[index : index + 1],
                    history_keys,
                    history_values,
                    scale=scale,
                    mask=effective,
                    sinks=sinks,
                    softcap=softcap,
                )
            )
        return mx.concatenate(attended)
