from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.cache import (
    FixedKVCache,
    KVCache,
    LayerCache,
    Layout,
    Rows,
    Snapshot,
)

type LatentStore = MLACache | FixedMLACache | BatchedMLACache
type NgramStore = NgramCache | FixedNgramCache | BatchedNgramCache
type LongcatLayer = LayerCache | BatchedMLACache | BatchedNgramCache
"""What a layer of this family's cache can be — a solo cache, or one of the ragged adapters
the batch path reads it through."""

_MIXED = "a batched layer mixes {kind} with another cache kind"


class _Batched[CacheT: LayerCache](LayerCache):
    """The rows of one ragged layer, held as they are; each adapter answers `offset` itself."""

    def __init__(self, caches: Sequence[CacheT]) -> None:
        self._caches = tuple(caches)

    @property
    def sequences(self) -> tuple[CacheT, ...]:
        return self._caches


class NgramCache(LayerCache):
    """The last ``n-1`` token ids, carried across the prefill→decode boundary."""

    def __init__(self, n: int, eos: int) -> None:
        super().__init__()
        self._context: mx.array | None = None
        self._n = n
        self._eos = eos

    def fetch_and_update(self, ids: mx.array) -> mx.array:
        """Return the full context (cached + new ids) and keep the last ``n-1``."""
        if self._context is not None:
            context = mx.concatenate([self._context, ids], axis=-1)
        else:
            context = ids
        keep = max(0, context.shape[-1] - self._n + 1)
        self._context = context[..., keep:]
        self.offset += ids.shape[-1]
        return context

    @property
    def is_fixable(self) -> bool:
        """Only where id 0 is not the eos. The fixed context is left-padded with zeros, and
        the eager shift's eos window counts the ids it sees — a zero that also means eos
        would zero a column the growing path leaves alone."""
        return self._eos != 0

    def fixed(self, capacity: int) -> "FixedNgramCache":
        del capacity
        if self._eos == 0:
            raise ValueError("an n-gram context padded with zeros cannot carry eos id 0")
        return FixedNgramCache(_padded(self._context, self._n - 1), self.offset)

    @property
    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return 0 if self._context is None else self._context.nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return () if self._context is None else (self._context,)

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        context = self._context

        def restore() -> None:
            parent()
            self._context = context

        return restore

    def trim(self, length: int) -> None:
        kept = min(self.offset, length)
        dropped = self.offset - kept
        self.offset = kept
        if dropped and self._context is not None:
            # The context is the tail of the history; rewinding the offset without cutting it
            # leaves the n-gram conditioned on ids the sequence no longer contains.
            remaining = self._context.shape[-1] - dropped
            self._context = None if remaining <= 0 else self._context[..., :remaining]

    @property
    def layout(self) -> Mapping[str, Layout]:
        """State, not history: what it holds is the last `n-1` ids of the sequence, which is
        a window and not a record — the ids before it are gone, and the next embedder reads
        exactly this. Tens of bytes, so the span it rides on costs nothing to keep."""
        return {"context": Snapshot()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        return {} if self._context is None else {"context": self._context}

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self._context = tensors.get("context")

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedNgramCache":
        return BatchedNgramCache(_rows(NgramCache, rows))


class FixedNgramCache(LayerCache):
    """The same ``n-1`` ids at a width the graph can hold.

    The growing cache keeps whatever it has (nothing at all before the first token); this
    one always keeps `n-1` columns, left-padded with zeros. That is what makes the step a
    single shape: the context goes in, `concat(context, ids)` comes out `n` wide, and what
    stays behind is that concatenation minus its first column.

    The pad is not a difference in what the embedder reads. `_shift_right_ignore_eos` pads
    with zeros itself, and every column it derives for the one row being embedded is at a
    fixed distance from the end — so a shorter history reads the same zeros either way.
    The one thing that would not survive is id 0 also being the eos, which `NgramCache.fixed`
    refuses.
    """

    offset: int

    def __init__(self, context: mx.array, position: int) -> None:
        super().__init__()
        self.offset = position
        self.state = [context]

    def fetch_and_update(self, ids: mx.array) -> mx.array:
        """The full context out, its tail kept — the growing cache's `fetch_and_update` with
        the widths frozen. The offset is left alone: what advances it is the caller that ran
        the step, and inside a trace this body runs once."""
        context = mx.concatenate([self.state[0], ids], axis=-1)
        self.state[0] = context[..., 1:]
        return context

    @property
    def is_fixable(self) -> bool:
        return True

    def fixed(self, capacity: int) -> "FixedNgramCache":
        """Already fixed, and at a width `capacity` has no say in: what it holds is `n-1`
        ids and a longer buffer would only be columns the embedder never reads."""
        del capacity
        return self

    @property
    def containers(self) -> list[mx.array] | None:
        return self.state

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def is_replayable(self) -> bool:
        """No: the ids the ring shifted out are gone, and the base's `checkpoint()` captures
        the offset alone."""
        return False

    def checkpoints(self, rows: int) -> bool:
        del rows
        return False

    @property
    def nbytes(self) -> int:
        return self.state[0].nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return (self.state[0],)

    @property
    def layout(self) -> Mapping[str, Layout]:
        return {"context": Snapshot()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        del start, stop
        return {"context": self.state[0]}

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self.state[0] = _padded(tensors.get("context"), self.state[0].shape[-1])

    def trim(self, length: int) -> None:
        raise NotImplementedError("a compiled n-gram context cannot be rewound")


def _padded(context: mx.array | None, width: int) -> mx.array:
    """The last `width` ids, left-padded with zeros — the shape a trace holds."""
    if context is None:
        return mx.zeros((1, width), dtype=mx.int64)
    held = context[..., -width:].astype(mx.int64)
    missing = width - held.shape[-1]
    if missing <= 0:
        return held
    pad = mx.zeros((*held.shape[:-1], missing), dtype=mx.int64)
    return mx.concatenate([pad, held], axis=-1)


class BatchedNgramCache(_Batched[NgramCache]):
    """N `NgramCache`s as one: each row's context concatenated along the batch axis.

    In regime every row holds exactly ``n-1`` ids, so the stack is exact. Rows that disagree
    about how much context they carry are a batch no embedder can read — the same refusal
    `batching._stack` makes for a recurrent batch mixing prefilled and empty rows.
    """

    @property
    def offset(self) -> int:
        return self._caches[0].offset

    @offset.setter
    def offset(self, value: int) -> None:
        # The rows are ragged, so there is no batch offset to assign: row 0 stands for the
        # batch only in how far it moved, and what propagates is that delta, not the value —
        # the same rule as `BatchedConvCache.offset`.
        advance = value - self._caches[0].offset
        for cache in self._caches:
            cache.offset += advance

    def fetch_and_update(self, ids: mx.array) -> mx.array:
        contexts = [
            cache.fetch_and_update(ids[index : index + 1])
            for index, cache in enumerate(self._caches)
        ]
        widths = {context.shape[-1] for context in contexts}
        if len(widths) != 1:
            raise ValueError("an n-gram batch mixes rows of different context lengths")
        return mx.concatenate(contexts)


class MLACache(KVCache):
    """The compressed latent (``kv_lora_rank``) + decoupled ``k_pe``
    (``qk_rope_head_dim``) per sublayer — 576 elements/token, not full K/V.

    `KVCache`'s storage with `keys` reading as the latent and `values` as `k_pe`: both grow
    on axis 2, one row per token, and nothing about how they are written differs. What
    differs is the read — the absorbed step attends `latent` against itself with the `k_pe`
    scores folded into the mask — which lives in the layer, not here. Only the names the
    spans are filed under are this class's own.
    """

    @property
    def is_fixable(self) -> bool:
        """Yes, where the base refuses its own subclasses: the two buffers are the rows in
        absolute order, which is what `FixedKVCache` holds."""
        return self._keys is not None and self._values is not None

    def fixed(self, capacity: int) -> "FixedMLACache":
        """`FixedKVCache.promote` sizes each buffer from its own tensor, which is what this
        cache needs — `kv_lora_rank` against `qk_rope_head_dim`."""
        return FixedMLACache.promote(self, capacity)

    @property
    def layout(self) -> Mapping[str, Layout]:
        """`KVCache`'s answer under this family's names: both tensors grow on axis 2, one
        row per token, and spans of them concatenate the same way."""
        return {"latent": Rows(), "k_pe": Rows()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        if self._keys is None or self._values is None:
            return {}
        return {
            "latent": self._keys[..., start:stop, :],
            "k_pe": self._values[..., start:stop, :],
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self._keys = tensors.get("latent")
        self._values = tensors.get("k_pe")

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedMLACache":
        return BatchedMLACache(_latents(rows))


class FixedMLACache(FixedKVCache):
    """`MLACache` promoted: the same two buffers at a fixed capacity, with the family's
    read and the family's span names kept.

    A plain `FixedKVCache` would be the right storage and the wrong layer — its `batched`
    hands back the dense adapter, which attends `keys` against `values`. Regrow is
    inherited: `core.cache.regrow` shapes each buffer from its own tensor and returns the
    cache's own class.
    """

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedMLACache":
        return BatchedMLACache(_latents(rows))

    @property
    def layout(self) -> Mapping[str, Layout]:
        return {"latent": Rows(), "k_pe": Rows()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        return {
            "latent": self.state[0][..., start:stop, :],
            "k_pe": self.state[1][..., start:stop, :],
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        latent, k_pe = tensors.get("latent"), tensors.get("k_pe")
        if latent is not None and k_pe is not None:
            self.state[0][..., :offset, :] = latent
            self.state[1][..., :offset, :] = k_pe
        self.state[2] = mx.array([offset], dtype=mx.int32)
        self.offset = offset


def _grown(buffer: mx.array, rows: int, capacity: int) -> mx.array:
    shape = list(buffer.shape)
    shape[2] = capacity
    grown = mx.zeros(shape, dtype=buffer.dtype)
    grown[..., :rows, :] = buffer[..., :rows, :]
    return grown


type LatentRow = MLACache | FixedMLACache
"""One sublayer's latent history, growing or promoted. A batch may hold either."""


class BatchedMLACache(_Batched[LatentRow]):
    """N `MLACache`s as one ragged latent batch.

    `BatchedKVCache`'s shape over two tensors instead of keys/values, and without an
    `attend`: MLA's absorbed decode folds the decoupled `k_pe` scores into the mask its
    latent attention reads, so the layer takes the rows back and attends each one itself.
    """

    @property
    def offset(self) -> mx.array:
        """Per-row positions, read where the row keeps them: a promoted row's lives in the
        graph and its `offset` is whatever the trace was built with."""
        return mx.concatenate(
            [
                cache.position
                if isinstance(cache, FixedMLACache)
                else mx.array([cache.offset], dtype=mx.int32)
                for cache in self._caches
            ]
        )

    @property
    def span(self) -> int:
        """The longest row's."""
        return max(cache.span for cache in self._caches)

    @property
    def materialized_kv_bytes(self) -> int:
        return 0

    def update_rows(
        self, latent: mx.array, k_pe: mx.array
    ) -> list[tuple[mx.array, mx.array]]:
        """Each row's own history, written and read one row at a time.

        Not `update_and_fetch`: that hands back one dense history, and these rows hold
        histories of different lengths with nothing past the projections shared."""
        return [
            cache.update_and_fetch(latent[index : index + 1], k_pe[index : index + 1])
            for index, cache in enumerate(self._caches)
        ]


def _rows[CacheT: LayerCache](kind: type[CacheT], rows: Sequence[LayerCache]) -> list[CacheT]:
    caches: list[CacheT] = []
    for row in rows:
        if not isinstance(row, kind):
            raise TypeError(_MIXED.format(kind=type(row).__name__))
        caches.append(row)
    return caches


def _latents(rows: Sequence[LayerCache]) -> list[LatentRow]:
    caches: list[LatentRow] = []
    for row in rows:
        if not isinstance(row, MLACache | FixedMLACache):
            raise TypeError(_MIXED.format(kind=type(row).__name__))
        caches.append(row)
    return caches
