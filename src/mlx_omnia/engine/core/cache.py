"""Per-layer KV cache backed by a block-grown buffer.

Rows are written into a preallocated buffer, not concatenated: concat copies
`offset` rows per step (O(context) per token; measured 25% at 4k tokens in the
Swift port). Growth copies once per block of 256 rows.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx


@dataclass(frozen=True)
class Rows:
    """History: a span holds the rows its own tokens produced, and spans concatenate.

    `stride` is tokens per row — 1 for a KV buffer, the compression ratio for a pooled one —
    and it is what a span length has to be a multiple of, or a span would end in the middle
    of a row nobody can cut.

    `keep` is the suffix that suffices to resume: the sliding window of a layer whose mask
    proves the rows before it are never read again. Spans deeper than that are not fetched at
    all and come back as zeros of the right shape, which keeps every row at its absolute
    position — the only thing the mask and the rotary tables are written against.
    """

    axis: int = 2
    stride: int = 1
    keep: int | None = None


@dataclass(frozen=True)
class Snapshot:
    """State: the whole value as it stood on a span's boundary.

    It does not compose — resuming reads the last one and nothing before it — and it only
    exists while the layer is stopped on the boundary, because a recurrence in the middle of
    a forward has no state outside the kernel.
    """


type Layout = Rows | Snapshot


@runtime_checkable
class Layouts(Protocol):
    """A trunk that knows more about its own cache than the cache classes do.

    The one case is a sliding layer: `laguna` builds every layer as a plain `KVCache` and the
    window lives in the config, so the class cannot answer `keep` and the family can. `None`
    means the classes' own answers stand.
    """

    def cache_layouts(self) -> Sequence[Mapping[str, Layout]] | None: ...


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

    @property
    def is_replayable(self) -> bool:
        """Whether `checkpoint()` is a *true* restore point — whether running the layer again
        over the same rows lands on the state it would have had.

        The default is yes, because that is what `checkpoint()` below promises and what every
        layer honours whose writes only land past the offset or which reassigns its tensors
        rather than mutating them. A layer that rotates its buffer, or whose position lives
        somewhere the base does not capture, answers `False` — and it is a property and not a
        docstring because a caller that rewinds by replay has to be able to ask.

        Read over the whole list the way `is_trimmable` is: one layer that cannot restore
        makes the trunk's rewind a lie, and a lie here is a decode running off state that
        describes a sequence that never happened.
        """
        return True

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

    @property
    def rows(self) -> int:
        """How many rows this layer holds *now*.

        The same number as `offset` for every cache whose counter is a host-side int. For one
        whose position lives in the graph it is not: `offset` there is the value the decode
        was traced with and stays frozen, because a forward that read the live count would
        have to evaluate it and a compiled trace cannot. So the forward asks `offset` and
        whoever writes the cache out asks here, which costs one evaluation and is paid once
        per file rather than once per token.
        """
        return self.offset

    @property
    def signature(self) -> dict[str, object]:
        """What the bytes of `stored()` mean beyond their shapes.

        Empty for a cache holding rows as the model produced them. A cache that encodes them,
        or one that rotates them, answers with what it did — and that answer goes into the
        key of the file, which is what keeps a cache written under one policy from being read
        back by a trunk built under another. Inside the layer and not in a caller's config
        because the layer is the only thing that knows: the trunk decides the policy, and the
        engine that files the bytes never sees it.
        """
        return {}

    @property
    def layout(self) -> Mapping[str, Layout]:
        """How each tensor of `stored()` composes along the spans of one conversation.

        Empty for a layer with no bytes of its own — a shared reader, a stateless block in a
        hybrid's pattern — which is not the same as a layer that cannot be stored: it still
        takes the offset back, and a trunk restored short of one layer's offset is a cache
        that decodes fluently off state that never existed.
        """
        return {}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """This layer's tensors, named, for the absolute token range `[start, stop)`.

        The range is in tokens and never in rows, because the two are the same number only
        for a plain KV buffer: a pooled cache writes one row per `ratio` tokens, a compressed
        one splits the range between a dense head and packed codes, and a ring holds the last
        window of it rotated. Each class does that arithmetic itself — nothing generic can,
        and a caller that guessed would be reading another layer's rows.

        A `Snapshot` tensor ignores the range: it is the whole value, and it is only valid
        while the layer stands on `stop`.

        Named rather than positional because the bytes outlive the process that wrote them,
        and a tuple's order is not a contract anybody can read back.
        """
        return {}

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """The inverse, into a layer a model has just made: the tensors of one conversation's
        spans, already composed, and the position they leave the layer at.

        The base takes the offset and nothing else, which is the whole of what a layer with no
        tensors has to come back to."""
        self.offset = offset

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

    @property
    def layout(self) -> Mapping[str, Layout]:
        return {"window": Snapshot()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        return {} if self.window is None else {"window": self.window}

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self.window = tensors.get("window")


class DeltaCache(ConvCache):
    """The DeltaNet's conv window plus its recurrent state.

    `is_trimmable` stays `False` and always will: a state trimmed to an earlier length cannot
    be reconstructed from itself, because the recurrence kept no history to subtract. What it
    can do is start over — `checkpoint()` captures the state and the window by reference, and
    both this and `ConvCache` reassign rather than mutate, so restoring the references is
    exact. That is what makes speculation possible here, by replay and not by trim.
    """

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

    @property
    def layout(self) -> Mapping[str, Layout]:
        return {"window": Snapshot(), "state": Snapshot()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        return super().stored(start, stop) | ({} if self.state is None else {"state": self.state})

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        super().restore(offset, tensors)
        self.state = tensors.get("state")


class KVCache(LayerCache):
    def __init__(self) -> None:
        super().__init__()
        self._keys: mx.array | None = None
        self._values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        needed = self.offset + keys.shape[2]
        key_buffer = reserve(self._keys, needed, keys)
        value_buffer = reserve(self._values, needed, values)
        key_buffer[..., self.offset : needed, :] = keys
        value_buffer[..., self.offset : needed, :] = values
        self._keys, self._values = key_buffer, value_buffer
        self.offset = needed
        return key_buffer[..., :needed, :], value_buffer[..., :needed, :]

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

    @property
    def layout(self) -> Mapping[str, Layout]:
        return {"keys": Rows(), "values": Rows()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """The rows of `[start, stop)`, not the buffers. `reserve` rounds up to the 256-row
        block, so handing them over whole would pass up to 255 rows of zero per layer on as
        state the model never wrote."""
        if self._keys is None or self._values is None:
            return {}
        return {
            "keys": self._keys[..., start:stop, :],
            "values": self._values[..., start:stop, :],
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """The buffers come back exactly `offset` long rather than block-padded, which
        `reserve` handles the way it handles any buffer too small for what is coming."""
        self.offset = offset
        self._keys = tensors.get("keys")
        self._values = tensors.get("values")

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
    def is_replayable(self) -> bool:
        """No: the position that moves is `state[2]`, inside the graph, and the base's
        `checkpoint()` captures `offset` — which the compiled decode never touches. A restore
        would report the old position over a buffer that already advanced."""
        return False

    @property
    def nbytes(self) -> int:
        return self.state[0].nbytes + self.state[1].nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(self.state)

    @property
    def rows(self) -> int:
        return int(self.position.item())

    @property
    def layout(self) -> Mapping[str, Layout]:
        """What it holds is a dense KV in absolute order — the fixed buffer is a `KVCache`
        that stopped growing, not another representation — so a growing cache reads its spans
        back unchanged. Which is the case worth having: a conversation whose decode was
        compiled would otherwise be the one conversation that keeps nothing."""
        return {"keys": Rows(), "values": Rows()}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """The rows of `[start, stop)` out of the fixed buffer. The caller cuts against
        `rows` and not `offset`: the position moved inside the compiled decode and only the
        state tensor followed it."""
        return {
            "keys": self.state[0][..., start:stop, :],
            "values": self.state[1][..., start:stop, :],
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """Into the buffer this cache was built with, which is where it differs from the
        growing one: the capacity is fixed at construction and the rows land inside it."""
        keys, values = tensors.get("keys"), tensors.get("values")
        if keys is not None and values is not None:
            self.state[0][..., :offset, :] = keys
            self.state[1][..., :offset, :] = values
        self.state[2] = mx.array([offset], dtype=mx.int32)
        self.offset = offset

    def trim(self, length: int) -> None:
        raise NotImplementedError("a compiled cache cannot be rewound")


HEADROOM = 768


def fit(offset: int) -> int:
    """The smallest 256-multiple holding `offset` plus generation headroom.

    Sized to the prompt rather than to a constant: a fixed buffer is read whole by every
    step's attention, so an oversized one is bytes on the decode's critical path."""
    return (offset + HEADROOM + 255) // 256 * 256


def regrow(cache: FixedKVCache, capacity: int) -> FixedKVCache:
    """A full fixed buffer copied into a larger one, rows and position preserved — what a
    generation that outgrows its capacity pays once per doubling, never per token."""
    keys, values = cache.fetch()
    rows = cache.rows
    shape = list(keys.shape)
    shape[2] = capacity
    grown_keys = mx.zeros(shape, dtype=keys.dtype)
    grown_values = mx.zeros(shape, dtype=values.dtype)
    grown_keys[..., :rows, :] = keys[..., :rows, :]
    grown_values[..., :rows, :] = values[..., :rows, :]
    mx.eval(grown_keys, grown_values)
    return FixedKVCache(grown_keys, grown_values, rows)


class FixedDeltaCache(DeltaCache):
    """A `DeltaCache` whose window and state live in a graph-visible container.

    The growing cache holds both as plain attributes, which `mx.compile` cannot see: an
    attribute rebound inside a traced function is rebound at trace time and never again.
    Here they sit in `graph`, the list a compiled decode passes as `inputs`/`outputs`, and
    the mixer's reads and writes go through properties over its slots — so the mamba layer
    runs unchanged over either cache.
    """

    def __init__(self, window: mx.array, state: mx.array, position: int) -> None:
        LayerCache.__init__(self)
        self.offset = position
        self.graph = [window, state]

    @classmethod
    def promote(cls, cache: DeltaCache) -> "FixedDeltaCache":
        """Copy a completed prefill cache into the compile-friendly form. Both tensors are
        set by then: a prefill that never ran has nothing worth compiling over."""
        window, state = cache.window, cache.state
        if window is None or state is None:
            raise ValueError("promoting a delta cache before its prefill filled it")
        return cls(window, state, cache.offset)

    @property
    def window(self) -> mx.array | None:
        return self.graph[0]

    @window.setter
    def window(self, value: mx.array | None) -> None:
        assert value is not None, "a fixed delta cache never clears its window"
        self.graph[0] = value

    @property
    def state(self) -> mx.array | None:
        return self.graph[1]

    @state.setter
    def state(self, value: mx.array | None) -> None:
        assert value is not None, "a fixed delta cache never clears its state"
        self.graph[1] = value


_BLOCK = 256


def reserve(buffer: mx.array | None, needed: int, like: mx.array) -> mx.array:
    """The block-grown buffer resizer ``KVCache`` writes through, and the entry caches
    defined outside this module (e.g. a latent KV cache) grow the same way with."""
    if buffer is not None and buffer.shape[2] >= needed:
        return buffer
    capacity = (needed + _BLOCK - 1) // _BLOCK * _BLOCK
    shape = list(like.shape)
    shape[2] = capacity
    grown = mx.zeros(shape, dtype=like.dtype)
    if buffer is not None:
        grown[..., : buffer.shape[2], :] = buffer
    return grown


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

    @property
    def layout(self) -> Mapping[str, Layout]:
        """Nothing of its own. What it reads is the storing layer's, republished on every
        forward before the block runs, so a resumed trunk gets the link back from the first
        step rather than from the store — and no span carries the same rows twice."""
        return {}


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
        """Copy the live tail of a growing cache into its absolute ring slots.

        Two concatenates and not a row-at-a-time loop: the rotation is one cut, and a Python
        loop over the window is a dispatch per row on the promotion of every long prompt."""
        source_keys, source_values = cache.fetch()
        rows = cache.offset
        start = max(0, rows - window)
        ring = cls(window)
        ring._keys = _rotated(source_keys[..., start:rows, :], rows, window)
        ring._values = _rotated(source_values[..., start:rows, :], rows, window)
        mx.eval(ring._keys, ring._values)
        ring.offset = rows
        ring.state = [ring._keys, ring._values, mx.array([rows], dtype=mx.int32)]
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
        if self._keys is None or self._values is None:
            shape = (keys.shape[0], keys.shape[1], self.window, keys.shape[3])
            self._keys = mx.zeros(shape, keys.dtype)
            self._values = mx.zeros(shape, values.dtype)
        key_buffer, value_buffer = self._keys, self._values
        assert key_buffer is not None and value_buffer is not None
        if length >= self.window:
            keys, values = keys[..., -self.window :, :], values[..., -self.window :, :]
            length = self.window
        start = self.offset % self.window
        head = min(length, self.window - start)
        key_buffer[..., start : start + head, :] = keys[..., :head, :]
        value_buffer[..., start : start + head, :] = values[..., :head, :]
        if head < length:
            tail = length - head
            key_buffer[..., :tail, :] = keys[..., head:, :]
            value_buffer[..., :tail, :] = values[..., head:, :]
        self.offset += length
        return key_buffer, value_buffer

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def is_replayable(self) -> bool:
        """No, and for a reason `trim` shares: slot `j` holds absolute position `j % window`,
        so a write wraps onto a row the ring still needs. The un-promoted path assigns into
        the buffer in place, which no restore of a reference can undo."""
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

    @property
    def rows(self) -> int:
        return int(self.position.item())

    @property
    def layout(self) -> Mapping[str, Layout]:
        """Rows, in absolute order, of the window it still holds.

        The rotation is not in what it hands over — `stored` undoes it and `restore` puts it
        back — so a ring and the growing cache the same sliding layer was prefilled as trade
        the same bytes. That is the case that matters: `laguna` prefills into `KVCache` and
        promotes to a ring for the compiled decode, and a rotation that leaked into the span
        would be a history read at the wrong absolute positions.

        `keep` is the window, because past it there is nothing to hand over: a ring that has
        wrapped has already dropped those rows, which is exactly what its reader's mask says
        it will never look at again.
        """
        return {"keys": Rows(keep=self.window), "values": Rows(keep=self.window)}

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        """The rows of `[start, stop)` in absolute order, unrotated.

        Only the window it still holds can be asked for; the rows before it were overwritten
        by their own successors, which is what a ring is. A caller reaching past that is a
        span longer than the window, refused at resolution rather than here."""
        keys, values = (
            (self.state[0], self.state[1]) if self.state is not None else (self._keys, self._values)
        )
        if keys is None or values is None:
            return {}
        rows = self.rows
        if start < rows - self.window or stop > rows:
            raise ValueError(
                f"a ring of {self.window} standing at {rows} cannot hand over [{start}, {stop})"
            )
        return {
            "keys": _unrotated(keys, start, stop, self.window),
            "values": _unrotated(values, start, stop, self.window),
        }

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """Rows in absolute order, back into the rotation, and into the un-promoted form:
        `state` stays `None`, because a promotion is a decision about compiling a decode and
        not a property of the rows. Whoever compiles again promotes again."""
        self.offset = offset
        keys, values = tensors.get("keys"), tensors.get("values")
        self._keys = None if keys is None else _rotated(keys, offset, self.window)
        self._values = None if values is None else _rotated(values, offset, self.window)
        self.state = None

    def reload(self, rows: int, keys: mx.array, values: mx.array) -> None:
        """Take another sequence's rotation whole, into the graph-visible state.

        The buffers are taken as they are rather than sliced or reordered: slot `j` already
        holds absolute position `j % window`, so the rotation *is* the state, and `rows` is
        only the position that goes with it. That is what separates this from `restore`,
        which takes rows in absolute order and has to put the rotation back.

        The slots are written **into** the list and the list is never replaced: a compiled
        decode captured this very object as its `inputs`/`outputs`, so a new one would leave
        the graph writing to a container the cache no longer reads. A ring that was never
        promoted has no such list, and gets one."""
        position = mx.array([rows], dtype=mx.int32)
        if self.state is None:
            self.state = [keys, values, position]
        else:
            self.state[0], self.state[1], self.state[2] = keys, values, position
        self._keys, self._values = keys, values
        self.offset = rows

    def trim(self, length: int) -> None:
        raise NotImplementedError("a ring keeps no history to rewind to")

    def fetch(self) -> tuple[mx.array, mx.array]:
        if self.state is not None:
            return self.state[0], self.state[1]
        assert self._keys is not None and self._values is not None
        return self._keys, self._values


def _rotated(rows: mx.array, offset: int, window: int) -> mx.array:
    """Rows in absolute order, ending at `offset`, into a ring of `window` slots.

    Always the ring's own buffer, never a view of the caller's: a ring that shared storage
    with the growing cache it was promoted from is a compiled decode writing through to a
    buffer somebody else still holds, which reads as a fluent wrong answer four tokens in.
    The full-window case is copied explicitly for that reason — a slice assignment that covers
    the whole array is elided, and what comes back is the source.

    One allocation and at most two writes, against the row-at-a-time loop this replaces: a
    dispatch per row of the window on the promotion of every long prompt.
    """
    count = rows.shape[2]
    head = (offset - count) % window
    if count == window and head == 0:
        return rows + mx.zeros_like(rows)
    shape = list(rows.shape)
    shape[2] = window
    ring = mx.zeros(shape, dtype=rows.dtype)
    split = min(count, window - head)
    ring[..., head : head + split, :] = rows[..., :split, :]
    if split < count:
        ring[..., : count - split, :] = rows[..., split:, :]
    return ring


def _unrotated(buffer: mx.array, start: int, stop: int, window: int) -> mx.array:
    """The ring's rows for absolute `[start, stop)`, back in order."""
    head = start % window
    count = stop - start
    if head + count <= window:
        return buffer[..., head : head + count, :]
    return mx.concatenate(
        [buffer[..., head:, :], buffer[..., : count - (window - head), :]], axis=2
    )


class Composite(LayerCache):
    """A layer whose state is several named caches advancing together.

    A hybrid's attention beside its indexer, a compressor beside its local window: the layer
    presents one offset and one set of tensors, and the parts answer for their own. Written
    once here because the delegation is the same every time — the families that carry it
    differ in which parts they hold and in nothing else, and four hand-written copies of it
    is four places for a part to be forgotten out of exactly one of them.
    """

    @property
    def parts(self) -> Mapping[str, LayerCache]:
        raise NotImplementedError("a composite cache names its parts")

    @property
    def is_replayable(self) -> bool:
        return all(part.is_replayable for part in self.parts.values())

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for part in self.parts.values())

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(tensor for part in self.parts.values() for tensor in part.tensors)

    @property
    def signature(self) -> dict[str, object]:
        held: dict[str, object] = {}
        for name, part in self.parts.items():
            held |= {f"{name}.{key}": value for key, value in part.signature.items()}
        return held

    @property
    def layout(self) -> Mapping[str, Layout]:
        held: dict[str, Layout] = {}
        for name, part in self.parts.items():
            held |= {f"{name}.{key}": value for key, value in part.layout.items()}
        return held

    def checkpoint(self) -> Callable[[], None]:
        restores = [part.checkpoint() for part in self.parts.values()]

        def restore() -> None:
            for undo in restores:
                undo()

        return restore

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        held: dict[str, mx.array] = {}
        for name, part in self.parts.items():
            held |= {f"{name}.{key}": value for key, value in part.stored(start, stop).items()}
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        for name, part in self.parts.items():
            prefix = f"{name}."
            part.restore(
                offset,
                {
                    key.removeprefix(prefix): value
                    for key, value in tensors.items()
                    if key.startswith(prefix)
                },
            )
