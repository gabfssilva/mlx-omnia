from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.engine.core.attend import AttentionMask
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, LayerCache


class LatentKVCache(KVCache):
    """The MLA cache: `keys` are the compressed latent and `values` are `k_pe`.

    Nothing about the storage differs from `KVCache` — what differs is the read. The
    absorbed step attends over `concat(latent, k_pe)` with `latent` as the values, which no
    generic adapter can express, so this class answers `batched` itself and hands back the
    family's own adapter.
    """

    @property
    def is_fixable(self) -> bool:
        """Yes, and the base says no only because it refuses its own subclasses — a cache
        that stores the same rows but reads them differently is not a plain buffer, and one
        that *can* be promoted has to say so itself. This one can: the two buffers are
        `[latent, k_pe]` in absolute order, which is what `FixedKVCache` holds, and the read
        that differs lives in the layer rather than in the storage."""
        return self._keys is not None and self._values is not None

    def fixed(self, capacity: int) -> "FixedLatentKVCache":
        """The fixed buffer that keeps the latent read. The widths differ between the two —
        `kv_lora_rank` against `qk_rope_head_dim` — and `FixedKVCache.promote` sizes each
        from its own tensor for exactly this case."""
        return FixedLatentKVCache.promote(self, capacity)

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedLatentKVCache":
        return BatchedLatentKVCache(_latents(rows))


class FixedLatentKVCache(FixedKVCache):
    """`LatentKVCache` promoted: the same two buffers at a fixed capacity, with the family's
    read kept.

    A plain `FixedKVCache` would be the right *storage* and the wrong *layer*: `batched`
    there hands back the dense adapter, which attends `keys` against `values` — and here the
    keys are `concat(latent, k_pe)` and the values are the latent alone. One method, and it
    is the reason this class exists rather than the base being used directly.
    """

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedLatentKVCache":
        return BatchedLatentKVCache(_latents(rows))


type LatentRow = LatentKVCache | FixedLatentKVCache
"""One sequence's latent history, growing or promoted. A batch may hold either."""


class BatchedLatentKVCache(LayerCache):
    """N latent caches as one ragged layer.

    The rows are ragged, so there is no dense history to hand back: `update_rows`
    writes each row's step into its own cache and returns the per-row pairs, which the
    layer attends one at a time.
    """

    def __init__(self, caches: Sequence[LatentRow]) -> None:
        self._caches = tuple(caches)

    @property
    def sequences(self) -> tuple[LatentRow, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        """Per-row positions, read where the row keeps them: a promoted row's live in the
        graph and its `offset` is whatever the trace was built with."""
        return mx.concatenate(
            [
                cache.position
                if isinstance(cache, FixedKVCache)
                else mx.array([cache.offset], dtype=mx.int32)
                for cache in self._caches
            ]
        )

    @property
    def span(self) -> int:
        """The longest row's."""
        return max(cache.span for cache in self._caches)

    def update_rows(
        self, latent: mx.array, k_pe: mx.array
    ) -> list[tuple[mx.array, mx.array, AttentionMask]]:
        """Each row's step into its own cache, and each row's history back with the mask it
        may be read under.

        Not `update_and_fetch`: that hands back one dense history, and these rows hold
        histories of different lengths with nothing past the projections shared. The mask is
        the row's own answer (`LayerCache.readable`) — `None` for a growing buffer that
        returned exactly what it wrote, the written columns for a promoted one that returned
        its whole capacity."""
        rows: list[tuple[mx.array, mx.array, AttentionMask]] = []
        for index, cache in enumerate(self._caches):
            history, history_pe = cache.update_and_fetch(
                latent[index : index + 1], k_pe[index : index + 1]
            )
            rows.append((history, history_pe, cache.readable(None, 1)))
        return rows


def _latents(rows: Sequence[LayerCache]) -> list[LatentRow]:
    latents: list[LatentRow] = []
    for row in rows:
        if not isinstance(row, LatentKVCache | FixedLatentKVCache):
            raise TypeError(f"a batched latent layer mixes {type(row).__name__} with it")
        latents.append(row)
    return latents
