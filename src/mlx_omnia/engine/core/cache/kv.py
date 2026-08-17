"""KV buffers: the one that grows, the one a graph can hold, and the ring a window keeps.

Rows are written into a preallocated buffer, not concatenated: concat copies `offset` rows
per step (O(context) per token; measured 25% at 4k tokens in the Swift port). Growth copies
once per block of 256 rows.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.attend import Attending, AttentionMask, dense_attention
from mlx_omnia.engine.core.cache.layer import LayerCache, row_mask, uniform
from mlx_omnia.engine.core.cache.layout import Layout, Rows


class KVCache(LayerCache):
    """A growing KV buffer, optionally read through a sliding window.

    `window` is the layer's, not the prefill's: the rows are kept whole either way — the
    mask is what makes a layer sliding — and what knowing the window buys is the two
    answers the class could not otherwise give. `layout` can say how deep a resume has to
    read, because past the window the mask proves the rows carry zero weight for every
    query that will ever run. And `fixed` can promote into a ring instead of a buffer,
    because a window's worth of rows is all a step will ever attend.

    Before this the window lived in a family's config and the family answered both
    questions itself, which is a model declaring something about its state rather than
    about its arithmetic.
    """

    offset: int

    def __init__(self, window: int | None = None) -> None:
        super().__init__()
        self.window = window
        self._keys: mx.array | None = None
        self._values: mx.array | None = None

    @property
    def is_fixable(self) -> bool:
        """The exact class, not the subclasses: what `fixed` hands back is a plain fixed
        buffer, and a subclass that stores the same rows but reads them differently — a
        latent layer attending over `concat(latent, k_pe)` — is not that. One that can be
        promoted says so itself."""
        return type(self) is KVCache and self._keys is not None and self._values is not None

    def fixed(self, capacity: int) -> "LayerCache":
        """A ring for a sliding layer, a fixed buffer for a full one.

        The ring is `window` rows and ignores `capacity`: what a sliding step attends is
        the window, and a buffer sized to the conversation would put rows on the decode's
        critical path that its own mask says are never read.
        """
        if self.window is not None:
            return RingKVCache.promote(self, self.window)
        return FixedKVCache.promote(self, capacity)

    def batched(self, rows: Sequence[LayerCache]) -> "LayerCache":
        return BatchedKVCache(uniform(rows, (KVCache, FixedKVCache, RingKVCache)))

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
        """`keep` is the window when the layer has one: past it the mask proves the rows
        carry zero weight for every query that will ever run, so a resume that skips those
        spans lands on the same logits."""
        return {"keys": Rows(keep=self.window), "values": Rows(keep=self.window)}

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

    def rewind(self, kept: int, position: int) -> None:
        """`LayerCache.rewind`. `trim` under another name: the rows past the offset are
        stale and the next update covers them."""
        del kept
        self.offset = position

    def fetch(self) -> tuple[mx.array, mx.array]:
        """The live K/V as written so far (rows 0..offset); rows past the offset are
        stale. A layer that publishes its full-length KV to a sharing layer reads
        through here instead of reaching into the private buffers."""
        assert self._keys is not None and self._values is not None
        return self._keys[..., : self.offset, :], self._values[..., : self.offset, :]


class FixedKVCache(LayerCache):
    """A fixed-capacity KV buffer with a graph-visible position."""

    offset: int

    def __init__(self, keys: mx.array, values: mx.array, position: int) -> None:
        super().__init__()
        self.offset = position
        self.state = [keys, values, mx.array([position], dtype=mx.int32)]

    @classmethod
    def promote(cls, cache: KVCache, capacity: int) -> Self:
        """Copy a growing cache into a fixed buffer suitable for ``mx.compile``."""
        keys, values = cache.fetch()
        if capacity < cache.offset:
            raise ValueError(f"capacity {capacity} is below cache length {cache.offset}")
        # Each buffer from its own tensor: a latent layer keeps keys and values of
        # different widths, and giving the values the keys' shape is a promotion whose
        # first read fails to broadcast.
        fixed_keys = mx.zeros(_capped(keys.shape, capacity), dtype=keys.dtype)
        fixed_values = mx.zeros(_capped(values.shape, capacity), dtype=values.dtype)
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

    def readable(self, mask: AttentionMask, queries: int, /) -> AttentionMask:
        """`LayerCache.readable`: the band `column <= the position this row landed on`.

        Fill and causality in one expression, which is why it serves a one-row decode and
        the `width + 1` rows of a verification alike: the `queries` rows just written end at
        `position`, so row *j* of them sits at `position - queries + j` and attends every
        column up to it — the rows before it in this same call included, and nothing past.
        Off `position`, which is a graph tensor here, so the band is rebuilt per step inside
        the trace instead of frozen at the first token.

        A string mask is replaced rather than intersected: `"causal"` aligns the queries to
        the *end* of the keys, and the end of a fixed buffer is capacity, not position.
        """
        columns = mx.arange(self.span)
        rows = self.position - queries + mx.arange(queries)[:, None]
        band = (columns[None, :] <= rows).reshape(1, 1, queries, self.span)
        return band if not isinstance(mask, mx.array) else band & mask

    def fetch(self) -> tuple[mx.array, mx.array]:
        return self.state[0], self.state[1]

    @property
    def span(self) -> int:
        return self.state[0].shape[2]

    @property
    def is_fixable(self) -> bool:
        return True

    def fixed(self, capacity: int) -> "LayerCache":
        return self if self.state[0].shape[2] == capacity else regrow(self, capacity)

    def batched(self, rows: Sequence[LayerCache]) -> "LayerCache":
        return BatchedKVCache(uniform(rows, (KVCache, FixedKVCache, RingKVCache)))

    @property
    def containers(self) -> list[mx.array] | None:
        return self.state

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

    def rewind(self, kept: int, position: int) -> None:
        """`LayerCache.rewind`. The position moves and the rows do not: what sits past it is
        stale and the next write covers it, exactly as `KVCache.trim` leaves things. Both
        counters move — `state[2]` is what the graph reads and `offset` is what the host
        does, and a rewind that moved only one of them would decode off a buffer that
        disagrees with itself."""
        del kept
        self.state[2] = mx.array([position], dtype=mx.int32)
        self.offset = position


HEADROOM = 768


def _capped(shape: tuple[int, ...], capacity: int) -> list[int]:
    """A buffer's shape with the row axis fixed at `capacity`."""
    fixed = list(shape)
    fixed[2] = capacity
    return fixed


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
    # Each buffer's own shape: the two need not agree past the row axis (a latent cache
    # pairs different widths; an index cache carries zero-width values).
    grown_keys = mx.zeros((*keys.shape[:2], capacity, keys.shape[3]), dtype=keys.dtype)
    grown_values = mx.zeros((*values.shape[:2], capacity, values.shape[3]), dtype=values.dtype)
    grown_keys[..., :rows, :] = keys[..., :rows, :]
    grown_values[..., :rows, :] = values[..., :rows, :]
    mx.eval(grown_keys, grown_values)
    # The cache's own class, not `FixedKVCache`: a subclass regrown through here keeps its
    # read, its span names and its batched adapter.
    return type(cache)(grown_keys, grown_values, rows)


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


class RingKVCache(LayerCache):
    """A sliding window as a fixed ring: `window` rows, written at `offset % window`.

    What it buys is not memory but *shape*. The growing cache hands attention a slice whose
    length changes every step, so the decode graph is rebuilt token by token and can never
    be compiled or fused. A ring's fetch is the same buffer at every step, which is the
    precondition for a one-dispatch attention that writes its own row.

    Only correct where every row of the window is attended: the reader must be a
    sliding-window layer whose mask is the window itself.
    """

    offset: int

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

    def readable(self, mask: AttentionMask, queries: int, /) -> AttentionMask:
        """`LayerCache.readable`: the slots this ring has written, and nothing else.

        Slot `j` holds the absolute position `a` with `a % window == j`, so while the ring
        is still short of a full window the filled slots are exactly the first `position` —
        and once it has wrapped every slot is filled, which is the same expression
        evaluating to all-true. One form for both regimes, which is what a trace needs.

        The caller's mask is *replaced* and never intersected, string or array alike: a
        positional band is written in absolute column order and these columns are rotated,
        so intersecting one would cut rows by an index that means nothing here. What makes
        that safe is what makes a ring legal at all — its reader is a sliding-window layer
        whose mask is the window itself.
        """
        assert queries == 1, "a ring is read one row at a time"
        del mask
        return (mx.arange(self.window) < self.position).reshape(1, 1, 1, self.window)

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
    def span(self) -> int:
        return self.window

    @property
    def is_fixable(self) -> bool:
        return self.state is not None or self._keys is not None

    def fixed(self, capacity: int) -> "LayerCache":
        """Into the graph-visible form, in place. The rotation already *is* what a step
        reads, so there is nothing to copy — only the position to move into a container the
        trace can write. A ring restored from a prefix arrives without one."""
        if self.state is None:
            keys, values = self.fetch()
            self.reload(self.offset, keys, values)
        return self

    def batched(self, rows: Sequence[LayerCache]) -> "LayerCache":
        return BatchedKVCache(uniform(rows, (KVCache, FixedKVCache, RingKVCache)))

    @property
    def containers(self) -> list[mx.array] | None:
        return self.state

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


class BatchedKVCache(LayerCache, Attending):
    """Present independent sequence caches as one ragged attention batch."""

    def __init__(self, caches: Sequence[KVCache | FixedKVCache | RingKVCache]) -> None:
        self._caches = tuple(caches)

    @property
    def sequences(self) -> tuple[KVCache | FixedKVCache | RingKVCache, ...]:
        """The per-sequence caches this layer stands for. Not `rows`, which on a cache is
        how many rows it holds — an adapter holds none of its own."""
        return self._caches

    @property
    def offset(self) -> mx.array:
        positions: list[mx.array] = []
        for cache in self._caches:
            match cache:
                case FixedKVCache() | RingKVCache():
                    positions.append(cache.position)
                case KVCache():
                    positions.append(mx.array([cache.offset], dtype=mx.int32))
        return mx.concatenate(positions)

    @property
    def span(self) -> int:
        """The longest row's."""
        return max(cache.span for cache in self._caches)

    @property
    def materialized_kv_bytes(self) -> int:
        return 0

    def attend(
        self,
        queries: mx.array,
        *,
        keys: mx.array,
        values: mx.array,
        scale: float,
        mask: AttentionMask,
        sinks: mx.array | None = None,
        softcap: float | None = None,
    ) -> mx.array:
        attended: list[mx.array] = []
        for index, cache in enumerate(self._caches):
            history_keys, history_values = cache.update_and_fetch(
                keys[index : index + 1], values[index : index + 1]
            )
            effective = cache.readable(
                row_mask(mask, index, len(self._caches), history_keys.shape[2]),
                queries.shape[2],
            )
            attended.append(
                dense_attention(
                    queries[index : index + 1],
                    history_keys,
                    history_values,
                    scale=scale,
                    mask=effective,
                    sinks=sinks,
                    softcap=softcap,
                )
            )
        return mx.concatenate(attended)
