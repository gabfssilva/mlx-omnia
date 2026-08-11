from collections.abc import Callable

import mlx.core as mx

from mlx_omnia.core.cache import KVCache, LayerCache


class DSACache(LayerCache):
    """One layer's two histories: the attention's keys/values and the indexer's keys.

    They advance together and rewind together, so the layer presents a single `offset`
    and a single `trim`; `generate.py` never sees that there are two.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention = KVCache()
        self.index = KVCache()

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
