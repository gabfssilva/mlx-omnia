from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.engine.batching import RaggedAdapter, RaggedBatchable
from mlx_omnia.engine.core.cache import KVCache, LayerCache


class LatentKVCache(KVCache, RaggedBatchable):
    """The MLA cache: `keys` are the compressed latent and `values` are `k_pe`.

    Nothing about the storage differs from `KVCache` — what differs is the read. The
    absorbed step attends over `concat(latent, k_pe)` with `latent` as the values, which no
    generic adapter can express, so this class answers `batched` itself and hands back the
    family's own adapter.
    """

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedLatentKVCache":
        latents: list[LatentKVCache] = []
        for row in rows:
            if not isinstance(row, LatentKVCache):
                raise TypeError(f"a batched latent layer mixes {type(row).__name__} with it")
            latents.append(row)
        return BatchedLatentKVCache(latents)


class BatchedLatentKVCache(RaggedAdapter):
    """N `LatentKVCache`s as one ragged layer.

    The rows are ragged, so there is no dense history to hand back: `update_and_fetch`
    writes each row's step into its own cache and returns the per-row pairs, which the
    layer attends one at a time.
    """

    def __init__(self, caches: Sequence[LatentKVCache]) -> None:
        self._caches = tuple(caches)

    @property
    def rows(self) -> tuple[LatentKVCache, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)

    def update_and_fetch(
        self, latent: mx.array, k_pe: mx.array
    ) -> list[tuple[mx.array, mx.array]]:
        return [
            cache.update_and_fetch(latent[index : index + 1], k_pe[index : index + 1])
            for index, cache in enumerate(self._caches)
        ]
