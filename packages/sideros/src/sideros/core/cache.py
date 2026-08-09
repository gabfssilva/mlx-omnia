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

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        """What this layer holds, for whoever needs the cache evaluated without evaluating
        what the same forward also produced — `core.prefill` is the caller, and the thing it
        is avoiding is the head. Same contract as `nbytes`: every subclass with a tensor of
        its own answers for it, one that only borrows another layer's answers nothing, and a
        layer holding only its offset has nothing to evaluate."""
        return ()

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

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return () if self.window is None else (self.window,)

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

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return super().tensors + (() if self.state is None else (self.state,))

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

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(buffer for buffer in (self._keys, self._values) if buffer is not None)

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


class FixedKVCache(LayerCache):
    """A fixed-capacity KV buffer with a graph-visible position."""

    def __init__(self, keys: mx.array, values: mx.array, position: int) -> None:
        super().__init__()
        self.offset = position
        self.state = [keys, values, mx.array([position], dtype=mx.int32)]

    @classmethod
    def promote(cls, cache: KVCache, capacity: int) -> "FixedKVCache":
        """Copy a growing cache into a fixed buffer suitable for ``mx.compile``."""
        keys, values = cache.fetch()
        if capacity < cache.offset:
            raise ValueError(f"capacity {capacity} is below cache length {cache.offset}")
        shape = list(keys.shape)
        shape[2] = capacity
        fixed_keys = mx.zeros(shape, dtype=keys.dtype)
        fixed_values = mx.zeros(shape, dtype=values.dtype)
        fixed_keys[..., : cache.offset, :] = keys
        fixed_values[..., : cache.offset, :] = values
        mx.eval(fixed_keys, fixed_values)
        return cls(fixed_keys, fixed_values, cache.offset)

    @property
    def position(self) -> mx.array:
        return self.state[2]

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.state[0] = mx.slice_update(self.state[0], keys, self.position, axes=(2,))
        self.state[1] = mx.slice_update(self.state[1], values, self.position, axes=(2,))
        self.state[2] = self.position + keys.shape[2]
        return self.state[0], self.state[1]

    def advance(self) -> None:
        """Advance after a fused kernel wrote the next row directly."""
        self.state[2] = self.position + 1

    def fetch(self) -> tuple[mx.array, mx.array]:
        return self.state[0], self.state[1]

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        return self.state[0].nbytes + self.state[1].nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(self.state)

    def trim(self, length: int) -> None:
        raise NotImplementedError("a compiled cache cannot be rewound")


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


class SharedKVReader(LayerCache):
    """A cache placeholder for KV-shared layers: no projection, no own buffer.

    Reads the full-length K,V from the storing layer's `KVCache` of the same type.
    `offset` tracks the position so the mask logic works, but the actual K,V come
    from `keys`/`values` set each step by the trunk.
    """

    def __init__(self) -> None:
        super().__init__()
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)


class RingKVCache(LayerCache):
    """A sliding window as a fixed ring: `window` rows, written at `offset % window`.

    What it buys is not memory but *shape*. The growing cache hands attention a slice whose
    length changes every step, so the decode graph is rebuilt token by token and can never
    be compiled or fused. A ring's fetch is the same buffer at every step, which is the
    precondition for a one-dispatch attention that writes its own row.

    Only correct where every row of the window is attended: the reader must be a
    sliding-window layer whose mask is the window itself.
    """

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = window
        self._keys: mx.array | None = None
        self._values: mx.array | None = None
        self.state: list[mx.array] | None = None

    @classmethod
    def promote(cls, cache: KVCache, window: int) -> "RingKVCache":
        """Copy the live tail of a growing cache into its absolute ring slots."""
        source_keys, source_values = cache.fetch()
        ring = cls(window)
        shape = list(source_keys.shape)
        shape[2] = window
        ring._keys = mx.zeros(shape, dtype=source_keys.dtype)
        ring._values = mx.zeros(shape, dtype=source_values.dtype)
        count = min(cache.offset, window)
        start = cache.offset - count
        for absolute in range(start, cache.offset):
            target = absolute % window
            ring._keys[..., target : target + 1, :] = source_keys[..., absolute : absolute + 1, :]
            ring._values[..., target : target + 1, :] = source_values[
                ..., absolute : absolute + 1, :
            ]
        mx.eval(ring._keys, ring._values)
        ring.offset = cache.offset
        ring.state = [ring._keys, ring._values, mx.array([cache.offset], dtype=mx.int32)]
        return ring

    @property
    def position(self) -> mx.array:
        if self.state is None:
            return mx.array([self.offset], dtype=mx.int32)
        return self.state[2]

    @property
    def write_index(self) -> int | mx.array:
        if self.state is None:
            return self.offset % self.window
        return self.position % self.window

    @property
    def populated(self) -> bool:
        """Every row of the window holds a real key: what the fused reader assumes, since
        it attends the whole ring without a mask."""
        return self._keys is not None and self.offset >= self.window

    def advance(self) -> None:
        """Account for a row a reader wrote into the ring itself."""
        if self.state is None:
            self.offset += 1
        else:
            self.state[2] = self.position + 1

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        length = keys.shape[2]
        if self.state is not None:
            # A promoted ring writes through its state: in-place assignment into a captured
            # buffer is not a graph the compiled decode can trace, and the row a reader that
            # is not the fused one appends still has to move the position.
            assert length == 1, "a promoted ring appends one row at a time"
            index = self.position % self.window
            self.state[0] = mx.slice_update(self.state[0], keys, index, axes=(2,))
            self.state[1] = mx.slice_update(self.state[1], values, index, axes=(2,))
            self.state[2] = self.position + length
            return self.state[0], self.state[1]
        if self._keys is None:
            shape = (keys.shape[0], keys.shape[1], self.window, keys.shape[3])
            self._keys = mx.zeros(shape, keys.dtype)
            self._values = mx.zeros(shape, values.dtype)
        assert self._values is not None
        if length >= self.window:
            keys, values = keys[..., -self.window :, :], values[..., -self.window :, :]
            length = self.window
        start = self.offset % self.window
        head = min(length, self.window - start)
        self._keys[..., start : start + head, :] = keys[..., :head, :]
        self._values[..., start : start + head, :] = values[..., :head, :]
        if head < length:
            tail = length - head
            self._keys[..., :tail, :] = keys[..., head:, :]
            self._values[..., :tail, :] = values[..., head:, :]
        self.offset += length
        return self._keys, self._values

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        if self.state is not None:
            return self.state[0].nbytes + self.state[1].nbytes
        return sum(b.nbytes for b in (self._keys, self._values) if b is not None)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        if self.state is not None:
            return tuple(self.state)
        return tuple(b for b in (self._keys, self._values) if b is not None)

    def trim(self, length: int) -> None:
        raise NotImplementedError("a ring keeps no history to rewind to")

    def fetch(self) -> tuple[mx.array, mx.array]:
        if self.state is not None:
            return self.state[0], self.state[1]
        assert self._keys is not None and self._values is not None
        return self._keys, self._values
