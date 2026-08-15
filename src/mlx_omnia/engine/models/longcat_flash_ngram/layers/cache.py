from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.batching import RaggedAdapter, RaggedBatchable
from mlx_omnia.engine.core.cache import LayerCache, Layout, Rows, Snapshot, reserve

type LatentStore = MLACache | BatchedMLACache
type NgramStore = NgramCache | BatchedNgramCache
type LongcatLayer = LayerCache | BatchedMLACache | BatchedNgramCache
"""What a layer of this family's cache can be — a solo cache, or one of the ragged adapters
the batch path reads it through."""

_MIXED = "a batched layer mixes {kind} with another cache kind"


class _Batched[CacheT: LayerCache](RaggedAdapter):
    """The rows of one ragged layer, held as they are; each adapter answers `offset` itself."""

    def __init__(self, caches: Sequence[CacheT]) -> None:
        self._caches = tuple(caches)

    @property
    def rows(self) -> tuple[CacheT, ...]:
        return self._caches


class NgramCache(LayerCache, RaggedBatchable):
    """The last ``n-1`` token ids, carried across the prefill→decode boundary."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._context: mx.array | None = None
        self._n = n

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


class MLACache(LayerCache, RaggedBatchable):
    """The compressed latent (``kv_lora_rank``) + decoupled ``k_pe``
    (``qk_rope_head_dim``) per sublayer — 576 elements/token, not full K/V.

    Growth and trim follow ``KVCache``: a block-grown buffer on axis 2, offset
    rewound by ``trim``. ``reserve`` (the public alias of ``KVCache``'s resizer)
    is reused from ``core.cache`` (the pattern is identical; the cache type is not).
    """

    def __init__(self) -> None:
        super().__init__()
        self._latent: mx.array | None = None
        self._k_pe: mx.array | None = None

    def update_and_fetch(
        self, latent: mx.array, k_pe: mx.array
    ) -> tuple[mx.array, mx.array]:
        needed = self.offset + latent.shape[2]
        latents = reserve(self._latent, needed, latent)
        pes = reserve(self._k_pe, needed, k_pe)
        self._latent, self._k_pe = latents, pes
        latents[..., self.offset : needed, :] = latent
        pes[..., self.offset : needed, :] = k_pe
        self.offset = needed
        return latents[..., :needed, :], pes[..., :needed, :]

    @property
    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return sum(buf.nbytes for buf in (self._latent, self._k_pe) if buf is not None)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(buf for buf in (self._latent, self._k_pe) if buf is not None)

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        latent = self._latent
        k_pe = self._k_pe

        def restore() -> None:
            parent()
            self._latent = latent
            self._k_pe = k_pe

        return restore

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)

    @property
    def layout(self) -> Mapping[str, Layout]:
        """`KVCache`'s answer over two tensors of its own widths: both grow on axis 2, one
        row per token, and spans of them concatenate the same way."""
        return {"latent": Rows(), "k_pe": Rows()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        if self._latent is None or self._k_pe is None:
            return {}
        return {
            "latent": self._latent[..., start:stop, :],
            "k_pe": self._k_pe[..., start:stop, :],
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self._latent = tensors.get("latent")
        self._k_pe = tensors.get("k_pe")

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedMLACache":
        return BatchedMLACache(_rows(MLACache, rows))


class BatchedMLACache(_Batched[MLACache]):
    """N `MLACache`s as one ragged latent batch.

    `BatchedKVCache`'s shape over two tensors instead of keys/values, and without an
    `attend`: MLA's absorbed decode folds the decoupled `k_pe` scores into the mask its
    latent attention reads, so the layer takes the rows back and attends each one itself.
    """

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)

    @property
    def materialized_kv_bytes(self) -> int:
        return 0

    def update_and_fetch(
        self, latent: mx.array, k_pe: mx.array
    ) -> list[tuple[mx.array, mx.array]]:
        """Each row's own history, written and read one row at a time."""
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
