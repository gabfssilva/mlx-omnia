"""Per-layer KV cache backed by a block-grown buffer.

Rows are written into a preallocated buffer, not concatenated: concat copies
`offset` rows per step (O(context) per token; measured 25% at 4k tokens in the
Swift port). Growth copies once per block of 256 rows.
"""

import mlx.core as mx

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


class LayerCache:
    """Per-layer cache contract: `offset` counts rows written; `trim` rewinds only
    when the layer keeps enough history — check `is_trimmable` first."""

    def __init__(self) -> None:
        self.offset = 0

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, length: int) -> None:
        raise NotImplementedError("this cache keeps no history to rewind to")


class ConvCache(LayerCache):
    """The `kernel - 1` row window of a causal short conv."""

    def __init__(self) -> None:
        super().__init__()
        self.window: mx.array | None = None


class DeltaCache(ConvCache):
    """The DeltaNet's conv window plus its recurrent state; a trimmed state cannot be
    reconstructed, so speculative decoding is off for this architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.state: mx.array | None = None


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

    def trim(self, length: int) -> None:
        """Rewind to the first `length` positions; the buffer stays, rows past the
        offset are stale and overwritten by the next update."""
        self.offset = min(self.offset, length)
