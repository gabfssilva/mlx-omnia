from collections.abc import Callable, Mapping

import mlx.core as mx

from mlx_omnia.engine.core.cache import DeltaCache, KVCache, LayerCache


def _under(prefix: str, tensors: Mapping[str, mx.array]) -> dict[str, mx.array]:
    return {
        name.removeprefix(prefix): tensor
        for name, tensor in tensors.items()
        if name.startswith(prefix)
    }


class FalconH1LayerCache(LayerCache):
    """Per-layer composite: the Mamba2 conv window + SSM state (a ``DeltaCache``)
    AND the attention KV cache. The SSM state ``[B, H, Dh, Ds]`` is fp32 and
    cannot be rewound (a trimmed recurrent state is unreconstructable), so
    speculative decoding is off for this architecture — same disclaimer as
    ``DeltaCache``."""

    def __init__(self) -> None:
        self.mamba = DeltaCache()
        self.kv = KVCache()
        # After the two, because the base writes `offset` and the property below forwards it.
        super().__init__()

    @property
    def offset(self) -> int:
        """The rows the two halves hold. They advance together — the block feeds every token
        through both — and this composite keeps none of its own, so the count has to be read
        off one of them: the attribute the base class writes is never advanced by anything
        here, and every reader of the contract from outside was seeing an empty cache."""
        return self.kv.offset

    @offset.setter
    def offset(self, rows: int) -> None:
        self.kv.offset = rows
        self.mamba.offset = rows

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        return self.mamba.nbytes + self.kv.nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return self.mamba.tensors + self.kv.tensors

    @property
    def is_storable(self) -> bool:
        """Both halves answer for themselves, so the composite does too. Without this it
        inherited the base's `False`, and a trunk that cannot be written out is also a trunk
        `generate` keeps no prompt boundary for — this architecture rebuilt every conversation
        from nothing, at a whole prefill per turn."""
        return self.mamba.is_storable and self.kv.is_storable

    def stored(self) -> dict[str, mx.array]:
        return {f"mamba.{name}": tensor for name, tensor in self.mamba.stored().items()} | {
            f"kv.{name}": tensor for name, tensor in self.kv.stored().items()
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """One offset for both, which is what `offset` above says: they never disagree."""
        self.mamba.restore(offset, _under("mamba.", tensors))
        self.kv.restore(offset, _under("kv.", tensors))

    def checkpoint(self) -> Callable[[], None]:
        mamba_restore = self.mamba.checkpoint()
        kv_restore = self.kv.checkpoint()
        offset = self.offset

        def restore() -> None:
            mamba_restore()
            kv_restore()
            self.offset = offset

        return restore
