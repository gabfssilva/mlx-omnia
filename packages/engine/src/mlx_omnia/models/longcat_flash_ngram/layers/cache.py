from collections.abc import Callable

import mlx.core as mx

from mlx_omnia.core.cache import LayerCache, reserve


class NgramCache(LayerCache):
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
        self.offset = min(self.offset, length)


class MLACache(LayerCache):
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
        self._latent = reserve(self._latent, needed, latent)
        self._k_pe = reserve(self._k_pe, needed, k_pe)
        self._latent[..., self.offset : needed, :] = latent
        self._k_pe[..., self.offset : needed, :] = k_pe
        self.offset = needed
        return self._latent[..., :needed, :], self._k_pe[..., :needed, :]

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
