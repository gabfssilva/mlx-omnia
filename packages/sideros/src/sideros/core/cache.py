"""Per-layer KV cache backed by a block-grown buffer.

Rows are written into a preallocated buffer, not concatenated: concat copies
`offset` rows per step (O(context) per token; measured 25% at 4k tokens in the
Swift port). Growth copies once per block of 256 rows.
"""

from collections.abc import Callable

import mlx.core as mx


class LayerCache:
    """Per-layer cache contract: `offset` counts rows written; `trim` rewinds only
    when the layer keeps enough history — check `is_trimmable` first."""

    def __init__(self) -> None:
        self.offset = 0

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        """What this layer holds, for whoever budgets caches that outlive a request. Every
        subclass with a tensor in it answers for its own; a layer holding only its offset
        weighs nothing."""
        return 0

    def checkpoint(self) -> Callable[[], None]:
        """A restore point at a call boundary, for a replay that runs the layer again over
        the same input. Unlike `trim`, this needs no kept history: the base restores the
        offset alone, exact for a cache whose writes only land past it (KVCache overwrites
        the stale rows on the next update); a subclass that reassigns tensors restores
        those references too."""
        offset = self.offset

        def restore() -> None:
            self.offset = offset

        return restore

    def trim(self, length: int) -> None:
        raise NotImplementedError("this cache keeps no history to rewind to")


class ConvCache(LayerCache):
    """The `kernel - 1` row window of a causal short conv."""

    def __init__(self) -> None:
        super().__init__()
        self.window: mx.array | None = None

    @property
    def nbytes(self) -> int:
        return 0 if self.window is None else self.window.nbytes

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        window = self.window

        def restore() -> None:
            parent()
            self.window = window

        return restore


class DeltaCache(ConvCache):
    """The DeltaNet's conv window plus its recurrent state; a trimmed state cannot be
    reconstructed, so speculative decoding is off for this architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.state: mx.array | None = None

    @property
    def nbytes(self) -> int:
        return super().nbytes + (0 if self.state is None else self.state.nbytes)

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        state = self.state

        def restore() -> None:
            parent()
            self.state = state

        return restore


class KVCache(LayerCache):
    def __init__(self) -> None:
        super().__init__()
        self._keys: mx.array | None = None
        self._values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        needed = self.offset + keys.shape[2]
        self._keys = _reserving(self._keys, needed, keys)
        self._values = _reserving(self._values, needed, values)
        self._keys[..., self.offset : needed, :] = keys
        self._values[..., self.offset : needed, :] = values
        self.offset = needed
        return self._keys[..., :needed, :], self._values[..., :needed, :]

    @property
    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        """The buffers, not the rows in use: what a stored cache costs the budget is what
        it has reserved, and growth rounds up to the block."""
        return sum(buffer.nbytes for buffer in (self._keys, self._values) if buffer is not None)

    def trim(self, length: int) -> None:
        """Rewind to the first `length` positions; the buffer stays, rows past the
        offset are stale and overwritten by the next update."""
        self.offset = min(self.offset, length)

    def fetch(self) -> tuple[mx.array, mx.array]:
        """The live K/V as written so far (rows 0..offset); rows past the offset are
        stale. A layer that publishes its full-length KV to a sharing layer reads
        through here instead of reaching into the private buffers."""
        assert self._keys is not None and self._values is not None
        return self._keys[..., : self.offset, :], self._values[..., : self.offset, :]


_BLOCK = 256


def _reserving(buffer: mx.array | None, needed: int, like: mx.array) -> mx.array:
    if buffer is not None and buffer.shape[2] >= needed:
        return buffer
    capacity = (needed + _BLOCK - 1) // _BLOCK * _BLOCK
    shape = list(like.shape)
    shape[2] = capacity
    grown = mx.zeros(shape, dtype=like.dtype)
    if buffer is not None:
        grown[..., : buffer.shape[2], :] = buffer
    return grown


def reserve(buffer: mx.array | None, needed: int, like: mx.array) -> mx.array:
    """Public entry to the block-grown buffer resizer, for caches defined outside
    this module (e.g. a latent KV cache) that grow the same way ``KVCache`` does."""
    return _reserving(buffer, needed, like)
