from collections.abc import Callable

import mlx.core as mx

from sideros.core.cache import DeltaCache, KVCache, LayerCache


class FalconH1LayerCache(LayerCache):
    """Per-layer composite: the Mamba2 conv window + SSM state (a ``DeltaCache``)
    AND the attention KV cache. The SSM state ``[B, H, Dh, Ds]`` is fp32 and
    cannot be rewound (a trimmed recurrent state is unreconstructable), so
    speculative decoding is off for this architecture — same disclaimer as
    ``DeltaCache``."""

    def __init__(self) -> None:
        super().__init__()
        self.mamba = DeltaCache()
        self.kv = KVCache()

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        return self.mamba.nbytes + self.kv.nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return self.mamba.tensors + self.kv.tensors

    def checkpoint(self) -> Callable[[], None]:
        mamba_restore = self.mamba.checkpoint()
        kv_restore = self.kv.checkpoint()
        offset = self.offset

        def restore() -> None:
            mamba_restore()
            kv_restore()
            self.offset = offset

        return restore
