from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.cache import (
    Composite,
    FixedKVCache,
    KVCache,
    LayerCache,
    Layout,
    Rows,
    Snapshot,
    regrow,
    reserve,
)
from mlx_omnia.engine.models.deepseek_v4.config import LOCAL, OVERLAP


class PoolCache(LayerCache):
    """One compressor's history: the pooled rows, plus the tail of tokens that has not
    completed a window yet.

    The pooled rows grow into a block-reserved buffer like `KVCache`'s (one row per
    `ratio` tokens, so the growth is rare). Nothing here rewinds: the tail buffer and the
    overlap carry are consumed as they are written, and a trimmed window cannot be pooled
    again from rows the cache no longer holds — the same reason DeltaNet's state closes
    speculation.
    """

    def __init__(self, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.tail_kv: mx.array | None = None
        self.tail_gate: mx.array | None = None
        self.remainder = 0
        self.pooled: mx.array | None = None
        # Not `rows`: the base answers that name with the offset, and this counter is the
        # pooled rows written — one per `ratio` tokens, not one per token.
        self.pooled_rows = 0
        self.previous: tuple[mx.array, mx.array] | None = None

    @property
    def is_replayable(self) -> bool:
        """No. `checkpoint()` cannot capture the tail: a call that completes a window reads
        `tail[:remainder]` and then overwrites those same rows with what is left over, so a
        restore rewinds the counter onto rows the round already destroyed. A replay would
        pool a window out of the wrong tokens and say nothing."""
        return False

    @property
    def nbytes(self) -> int:
        buffers = (self.tail_kv, self.tail_gate, self.pooled)
        return sum(buffer.nbytes for buffer in buffers if buffer is not None)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        buffers = (self.tail_kv, self.tail_gate, self.pooled, *(self.previous or ()))
        return tuple(buffer for buffer in buffers if buffer is not None)

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        state = (self.remainder, self.pooled_rows, self.previous, self.pooled)

        def restore() -> None:
            parent()
            self.remainder, self.pooled_rows, self.previous, self.pooled = state

        return restore

    def accumulate(
        self, kv: mx.array, gate: mx.array, offset: int
    ) -> tuple[mx.array, mx.array, int]:
        """The rows that complete whole windows, and the absolute position of the first of
        them. What is left over is kept for the next call."""
        length = kv.shape[1]
        tail_kv, tail_gate = self.tail_kv, self.tail_gate
        if tail_kv is None or tail_gate is None:
            tail_kv = mx.zeros((1, self.ratio, kv.shape[-1]), dtype=kv.dtype)
            tail_gate = mx.zeros((1, self.ratio, gate.shape[-1]), dtype=gate.dtype)
            self.tail_kv, self.tail_gate = tail_kv, tail_gate

        total = length + self.remainder
        usable = total // self.ratio * self.ratio
        rest = total % self.ratio
        if usable:
            ready_kv = mx.concatenate(
                [tail_kv[:, : self.remainder], kv[:, : usable - self.remainder]], axis=1
            )
            ready_gate = mx.concatenate(
                [tail_gate[:, : self.remainder], gate[:, : usable - self.remainder]], axis=1
            )
            base = offset - self.remainder
            self.remainder = 0
        else:
            ready_kv, ready_gate, base = kv[:, :0], gate[:, :0], 0

        if rest:
            tail_kv[:, self.remainder : rest] = kv[:, -rest:]
            tail_gate[:, self.remainder : rest] = gate[:, -rest:]
        self.remainder = rest
        return ready_kv, ready_gate, base

    def append(self, pooled: mx.array) -> None:
        needed = self.pooled_rows + pooled.shape[2]
        buffer = reserve(self.pooled, needed, pooled)
        self.pooled = buffer
        buffer[..., self.pooled_rows : needed, :] = pooled
        self.pooled_rows = needed

    def fetch(self, head_dim: int, dtype: mx.Dtype) -> mx.array:
        if self.pooled is None:
            return mx.zeros((1, 1, 0, head_dim), dtype=dtype)
        return self.pooled[..., : self.pooled_rows, :]

    def mask(self, length: int, offset: int) -> mx.array | None:
        """Pooled row `i` is visible to the query at absolute position `p` while
        `i < (p + 1) // ratio`. At T=1 every row already in the cache passes."""
        if length == 1:
            return None
        rows = mx.arange(offset + 1, offset + length + 1).reshape(-1, 1)
        return mx.arange(self.pooled_rows).reshape(1, -1) < rows // self.ratio

    @property
    def layout(self) -> Mapping[str, Layout]:
        """The pooled rows compose at one row per `ratio` tokens; the overlap carry does not.


        The tail is neither, and it is nowhere: on a boundary that is a multiple of `ratio` —
        which the span is required to be — `remainder` is zero and the tail buffer holds
        nothing a next call will read. `pooled_rows` is `offset // ratio` and is recomputed
        on the way in rather than carried, for the same reason `QuantizedKVCache` recomputes
        its split: a second copy of a rule is a second rule, free to disagree.
        """
        held: dict[str, Layout] = {"pooled": Rows(stride=self.ratio)}
        if self.ratio == OVERLAP:
            # Only the overlapping ratio carries anything across a window: each of its windows
            # pools its own lane B with the previous one's lane A, so the last raw window is
            # state. Every other ratio pools a window out of its own tokens and keeps nothing
            # — and declaring a carry it never produces would be a trunk that claims an anchor
            # and hands over none, which is silent zero reuse for the whole model.
            held["carry.kv"] = Snapshot()
            held["carry.gate"] = Snapshot()
        return held

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        if start % self.ratio or stop % self.ratio:
            raise ValueError(
                f"a pool of {self.ratio} cannot cut [{start}, {stop}): a span has to close "
                "the window, or its last row is built from tokens on both sides of the cut"
            )
        held: dict[str, mx.array] = {}
        if self.pooled is not None:
            held["pooled"] = self.pooled[..., start // self.ratio : stop // self.ratio, :]
        if self.previous is not None:
            held["carry.kv"], held["carry.gate"] = self.previous
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        self.offset = offset
        self.remainder = 0
        self.pooled = tensors.get("pooled")
        self.pooled_rows = offset // self.ratio
        kv, gate = tensors.get("carry.kv"), tensors.get("carry.gate")
        self.previous = None if kv is None or gate is None else (kv, gate)


class FixedPoolCache(LayerCache):
    """A `PoolCache` in the shape a traced step can hold: a ring of raw rows, a
    capacity-sized pooled buffer, and one divided index writing both.

    The `if usable:` of the growing form does not survive a trace, so it is gone rather than
    scheduled around. Every step writes its raw row at `position % slots`, pools the whole
    window masked to the rows the position has reached, and writes the result at slot
    `position // ratio`. A window that is not complete yet writes a partial pool into the
    slot its completing step will overwrite, and `mask` hides that slot until the same step
    makes it visible — so what is read is only ever a pool over a full window, produced by
    the arithmetic the growing form's completing call runs.

    The ring is `ratio` rows wide for a pool that keeps nothing across a window, and two
    windows wide for the overlapping one, whose window pools its own lane B together with
    the previous window's lane A. Both index at `position % slots`, which for the two-window
    ring lands exactly at `ratio * (window % 2) + position % ratio`.

    The tensors live in the composite's one container list — `mx.compile` swaps arrays
    inside a captured container in place, so what the graph and the cache share has to be
    that list and not this object's attributes.
    """

    offset: int

    def __init__(self, state: list[mx.array], at: int, ratio: int, position: int) -> None:
        super().__init__()
        self.offset = position
        self.ratio = ratio
        self.overlap = ratio == OVERLAP
        self._state = state
        self._at = at

    @classmethod
    def promote(
        cls, cache: PoolCache, state: list[mx.array], capacity: int, position: int
    ) -> "FixedPoolCache":
        """Copy a growing pool into the ring and the capacity buffer, at `position`.

        The rows of the window in flight are the growing form's own tail, in the same order
        — slot `i` of the window holds the token at `window_start + i` in both — and the
        overlapping pool's carry is the previous raw window, dropped into the ring slot the
        next window's pool reads it from.
        """
        tail_kv, tail_gate = cache.tail_kv, cache.tail_gate
        if tail_kv is None or tail_gate is None:
            raise ValueError("a pool that never accumulated a row cannot be promoted")
        ratio = cache.ratio
        overlap = ratio == OVERLAP
        slots = ratio * (2 if overlap else 1)
        width = tail_kv.shape[-1]
        head_dim = width // 2 if overlap else width
        window, inside = divmod(position, ratio)
        rows = -(-capacity // ratio)
        if rows < window:
            raise ValueError(f"capacity {capacity} is below the {window} pooled rows held")

        ring_kv = mx.zeros((1, slots, width), dtype=tail_kv.dtype)
        ring_gate = mx.zeros((1, slots, width), dtype=tail_gate.dtype)
        base = ratio * (window % 2) if overlap else 0
        ring_kv[:, base : base + inside] = tail_kv[:, :inside]
        ring_gate[:, base : base + inside] = tail_gate[:, :inside]
        if overlap and cache.previous is not None:
            carry_kv, carry_gate = cache.previous
            before = ratio * ((window + 1) % 2)
            ring_kv[:, before : before + ratio] = carry_kv[:, 0]
            ring_gate[:, before : before + ratio] = carry_gate[:, 0]

        held = cache.pooled
        pooled = mx.zeros(
            (1, 1, rows, head_dim), dtype=held.dtype if held is not None else tail_kv.dtype
        )
        if held is not None:
            pooled[..., :window, :] = held[..., :window, :]
        mx.eval(ring_kv, ring_gate, pooled)
        at = len(state)
        state.extend((ring_kv, ring_gate, pooled))
        return cls(state, at, ratio, position)

    def regrown(self, state: list[mx.array], capacity: int, position: int) -> "FixedPoolCache":
        """This pool into a larger buffer: the ring is size-free and moves as it is, and only
        the pooled rows are copied across."""
        rows = -(-capacity // self.ratio)
        pooled = mx.zeros((1, 1, rows, self.pooled.shape[3]), dtype=self.pooled.dtype)
        kept = min(rows, self.pooled.shape[2])
        pooled[..., :kept, :] = self.pooled[..., :kept, :]
        ring_kv = self.ring_kv + mx.zeros_like(self.ring_kv)
        ring_gate = self.ring_gate + mx.zeros_like(self.ring_gate)
        mx.eval(ring_kv, ring_gate, pooled)
        at = len(state)
        state.extend((ring_kv, ring_gate, pooled))
        return FixedPoolCache(state, at, self.ratio, position)

    @property
    def slots(self) -> int:
        """Raw rows the ring holds: one window, or the two an overlapping pool spans."""
        return self.ratio * (2 if self.overlap else 1)

    @property
    def ring_kv(self) -> mx.array:
        return self._state[self._at]

    @property
    def ring_gate(self) -> mx.array:
        return self._state[self._at + 1]

    @property
    def pooled(self) -> mx.array:
        return self._state[self._at + 2]

    @property
    def rows(self) -> int:
        return self.pooled.shape[2]

    def write(self, kv: mx.array, gate: mx.array, position: mx.array) -> None:
        """This step's raw row into the ring, at the slot its position owns."""
        index = position % self.slots
        self._state[self._at] = mx.slice_update(self.ring_kv, kv, index, axes=(1,))
        self._state[self._at + 1] = mx.slice_update(self.ring_gate, gate, index, axes=(1,))

    def window(self, position: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """The rows this step pools, the gates that go with them, and which of them the
        position has actually reached.

        For the overlapping pool the rows are the previous window's lane A followed by this
        window's lane B — the layout the growing `_pool` builds by shifting whole windows —
        and the previous window is valid only once there has been one.
        """
        inside = mx.arange(self.ratio)
        here = inside <= position % self.ratio
        if not self.overlap:
            kv = self.ring_kv[:, None]
            gate = self.ring_gate[:, None]
            return kv, gate, here.reshape(1, 1, self.ratio, 1)
        half = self.ring_kv.shape[-1] // 2
        window = position // self.ratio
        current = self.ratio * (window % 2) + inside
        previous = self.ratio * ((window + 1) % 2) + inside
        kv = mx.concatenate(
            [
                mx.take(self.ring_kv, previous, axis=1)[..., :half],
                mx.take(self.ring_kv, current, axis=1)[..., half:],
            ],
            axis=1,
        )[:, None]
        gate = mx.concatenate(
            [
                mx.take(self.ring_gate, previous, axis=1)[..., :half],
                mx.take(self.ring_gate, current, axis=1)[..., half:],
            ],
            axis=1,
        )[:, None]
        before = mx.broadcast_to(window >= 1, (self.ratio,))
        return kv, gate, mx.concatenate([before, here]).reshape(1, 1, 2 * self.ratio, 1)

    def append(self, pooled: mx.array, position: mx.array) -> None:
        """This step's pooled row at the slot its window owns — every step, complete or
        not."""
        self._state[self._at + 2] = mx.slice_update(
            self.pooled, pooled, position // self.ratio, axes=(2,)
        )

    def fetch(self, head_dim: int, dtype: mx.Dtype) -> mx.array:
        del head_dim, dtype
        return self.pooled

    def mask(self, length: int, position: mx.array) -> mx.array:
        """`PoolCache.mask` off a graph tensor: row `i` is visible to the query at `p` while
        `i < (p + 1) // ratio`. The growing form's `None` shortcut at T=1 was the luxury of a
        buffer holding exactly the rows it had written; this one holds capacity, and the
        partial pool sitting in the window's own slot is what the rule hides."""
        assert length == 1, "a fixed pool is read one row at a time"
        return mx.arange(self.rows).reshape(1, -1) < (position + 1) // self.ratio

    @property
    def is_replayable(self) -> bool:
        """No, for `PoolCache`'s reason and one more: the position lives in the graph."""
        return False

    @property
    def is_fixable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return self.ring_kv.nbytes + self.ring_gate.nbytes + self.pooled.nbytes

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return (self.ring_kv, self.ring_gate, self.pooled)

    @property
    def layout(self) -> Mapping[str, Layout]:
        """`PoolCache.layout`: the same rows composing the same way. A conversation whose
        decode was compiled hands over what the growing form would have."""
        held: dict[str, Layout] = {"pooled": Rows(stride=self.ratio)}
        if self.overlap:
            held["carry.kv"] = Snapshot()
            held["carry.gate"] = Snapshot()
        return held

    def stored(self, start: int, stop: int) -> dict[str, mx.array]:
        if start % self.ratio or stop % self.ratio:
            raise ValueError(
                f"a pool of {self.ratio} cannot cut [{start}, {stop}): a span has to close "
                "the window, or its last row is built from tokens on both sides of the cut"
            )
        held: dict[str, mx.array] = {
            "pooled": self.pooled[..., start // self.ratio : stop // self.ratio, :]
        }
        if self.overlap:
            # The window before the one `stop` opens, which is what the growing form keeps as
            # `previous` — and at a boundary that closes a window, the ring's other half.
            before = self.ratio * ((stop // self.ratio + 1) % 2)
            held["carry.kv"] = self.ring_kv[:, None, before : before + self.ratio]
            held["carry.gate"] = self.ring_gate[:, None, before : before + self.ratio]
        return held

    def restore(self, offset: int, tensors: Mapping[str, mx.array]) -> None:
        """Into the buffers this cache was built with. The offset closes a window — `stored`
        refuses any other — so the ring's current half holds nothing a step will read."""
        self.offset = offset
        pooled = tensors.get("pooled")
        if pooled is not None:
            self.pooled[..., : offset // self.ratio, :] = pooled
        carry_kv, carry_gate = tensors.get("carry.kv"), tensors.get("carry.gate")
        if carry_kv is not None and carry_gate is not None:
            before = self.ratio * ((offset // self.ratio + 1) % 2)
            self.ring_kv[:, before : before + self.ratio] = carry_kv[:, 0]
            self.ring_gate[:, before : before + self.ratio] = carry_gate[:, 0]

    def rewind(self, kept: int, position: int) -> None:
        del kept
        self.offset = position


class DeepseekV4Cache(Composite):
    """One layer's histories: the local keys (K == V, one buffer), the compressor's pooled
    rows and — on the indexed layers — the indexer's own pooled rows. They advance
    together and the layer presents a single `offset`."""

    def __init__(self, ratio: int, indexed: bool) -> None:
        # Before `super().__init__()`: the base sets `offset`, and this class answers that
        # name with the local cache's.
        self.attention = KVCache()
        self.compressor = PoolCache(ratio) if ratio != LOCAL else None
        self.indexer = PoolCache(ratio) if indexed else None
        super().__init__()

    @property
    def parts(self) -> Mapping[str, LayerCache]:
        """The local window always, the two pools when this layer carries them. Absent and
        not empty: a layer with no compressor has no compressed rows to name, and a name
        that is sometimes empty is a name a resume would look for and not find."""
        held: dict[str, LayerCache] = {"attention": self.attention}
        if self.compressor is not None:
            held["compressor"] = self.compressor
        if self.indexer is not None:
            held["indexer"] = self.indexer
        return held

    @property
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value

    @property
    def is_fixable(self) -> bool:
        """The local window has to be promotable, and every pool has to have seen a row —
        the row is what names the widths and the dtypes its ring is built from."""
        pools = (self.compressor, self.indexer)
        return self.attention.is_fixable and all(
            pool.tail_kv is not None and pool.tail_gate is not None
            for pool in pools
            if pool is not None
        )

    def fixed(self, capacity: int) -> "LayerCache":
        return FixedDeepseekV4Cache.promote(self, capacity)

    @property
    def is_trimmable(self) -> bool:
        return self.compressor is None and self.attention.is_trimmable

    def trim(self, length: int) -> None:
        if self.compressor is not None:
            raise NotImplementedError("a pooled window cannot be recompressed after a trim")
        self.attention.trim(length)

    def batched(self, rows: Sequence[LayerCache]) -> "BatchedDeepseekV4Cache":
        """This layer's rows as one ragged batch. The pools' state is scalar *per row* —
        a partial tail, a remainder, a pooled length that differs between rows — so the
        adapter holds the rows and the forward walks them, rather than stacking anything."""
        held: list[DeepseekV4Cache] = []
        for row in rows:
            if not isinstance(row, DeepseekV4Cache):
                raise TypeError(f"a batched deepseek_v4 layer mixes in {type(row).__name__}")
            if (row.compressor is None) != (self.compressor is None) or (
                row.indexer is None
            ) != (self.indexer is None):
                raise TypeError("a batched deepseek_v4 layer mixes compress ratios")
            held.append(row)
        return BatchedDeepseekV4Cache(held)


class FixedDeepseekV4Cache(Composite):
    """`DeepseekV4Cache` in the shape a traced decode can hold: a `FixedKVCache` for the
    local window, a `FixedPoolCache` for each pool, and one container list under all of them.

    One list and not three, because `mx.compile` takes containers as its `inputs`/`outputs`
    and swaps the arrays inside them in place: a layer that handed over a freshly built list
    of its parts' lists would watch the graph write into the wrapper and leave the parts
    where they were. The local cache's own `state` is that list — it owns slots 0..2 and the
    pools take three each after it — so the position the whole layer advances by is the one
    `FixedKVCache` already moves.
    """

    def __init__(
        self,
        attention: FixedKVCache,
        compressor: FixedPoolCache | None,
        indexer: FixedPoolCache | None,
    ) -> None:
        # No `super().__init__()`: the base's is an assignment to `offset`, and this class
        # answers that name with the local cache's — which already stands where the prefill
        # left it. Zeroing it here would promote a full buffer onto position 0.
        self.attention = attention
        self.compressor = compressor
        self.indexer = indexer

    @classmethod
    def promote(cls, cache: DeepseekV4Cache, capacity: int) -> "FixedDeepseekV4Cache":
        position = cache.offset
        attention = FixedKVCache.promote(cache.attention, capacity)
        state = attention.state
        return cls(
            attention,
            None
            if cache.compressor is None
            else FixedPoolCache.promote(cache.compressor, state, capacity, position),
            None
            if cache.indexer is None
            else FixedPoolCache.promote(cache.indexer, state, capacity, position),
        )

    def fixed(self, capacity: int) -> "LayerCache":
        """Idempotent at the capacity it stands on, and a copy into larger buffers below it
        — the same doubling `FixedKVCache` pays once per regrow rather than per token."""
        if self.attention.span == capacity:
            return self
        position = self.attention.rows
        grown = regrow(self.attention, capacity)
        state = grown.state
        return FixedDeepseekV4Cache(
            grown,
            None if self.compressor is None else self.compressor.regrown(state, capacity, position),
            None if self.indexer is None else self.indexer.regrown(state, capacity, position),
        )

    @property
    def parts(self) -> Mapping[str, LayerCache]:
        held: dict[str, LayerCache] = {"attention": self.attention}
        if self.compressor is not None:
            held["compressor"] = self.compressor
        if self.indexer is not None:
            held["indexer"] = self.indexer
        return held

    @property
    def offset(self) -> int:
        return self.attention.offset

    @offset.setter
    def offset(self, value: int) -> None:
        self.attention.offset = value
        for pool in (self.compressor, self.indexer):
            if pool is not None:
                pool.offset = value

    @property
    def position(self) -> mx.array:
        """Where the layer stands, as the graph sees it. Read before the local window's
        update moves it — every rotation and every window this step builds is off this."""
        return self.attention.position

    @property
    def containers(self) -> list[mx.array] | None:
        return self.attention.state

    @property
    def is_fixable(self) -> bool:
        return True

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        """The one list, not the parts' views of it: the pools read their slots out of the
        local cache's `state`, and `Composite`'s walk would count every tensor twice."""
        return tuple(self.attention.state)

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.attention.state)

    @property
    def rows(self) -> int:
        return self.attention.rows

    @property
    def span(self) -> int:
        return self.attention.span

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, length: int) -> None:
        raise NotImplementedError("a compiled cache cannot be rewound")

    def rewind(self, kept: int, position: int) -> None:
        self.attention.rewind(kept, position)
        for pool in (self.compressor, self.indexer):
            if pool is not None:
                pool.rewind(kept, position)


type DeepseekV4Solo = DeepseekV4Cache | FixedDeepseekV4Cache
"""One sequence's layer cache: the growing form a prefill writes, or the fixed one a
compiled decode steps."""


class BatchedDeepseekV4Cache(LayerCache):
    """N `DeepseekV4Cache`s as one ragged layer: the rows, and where each one stands."""

    def __init__(self, caches: Sequence[DeepseekV4Cache]) -> None:
        self._caches = tuple(caches)

    @property
    def sequences(self) -> tuple[DeepseekV4Cache, ...]:
        return self._caches

    @property
    def offset(self) -> mx.array:
        return mx.array([cache.offset for cache in self._caches], dtype=mx.int32)

    @property
    def span(self) -> int:
        """The longest row's."""
        return max(cache.span for cache in self._caches)
