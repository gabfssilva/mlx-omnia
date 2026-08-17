"""The state a recurrent mixer keeps: a short conv's window, and the DeltaNet's own."""

from collections.abc import Callable, Mapping, Sequence

import mlx.core as mx

from mlx_omnia.engine.core.cache.layer import LayerCache, mixed, uniform
from mlx_omnia.engine.core.cache.layout import Layout, Snapshot


class ConvCache(LayerCache):
    """The `kernel - 1` row window of a causal short conv."""

    offset: int

    def __init__(self) -> None:
        super().__init__()
        self.window: mx.array | None = None

    @property
    def is_fixable(self) -> bool:
        return self.window is not None

    def checkpoints(self, rows: int) -> bool:
        """No. The window rolls and the recurrence consumes what it reads, so this form
        keeps none of the intermediate states a rewind would land on. `FixedDeltaCache` has
        slots for exactly that; nothing else here does."""
        del rows
        return False

    def fixed(self, capacity: int) -> "LayerCache":
        return FixedConvCache.promote(self)

    def batched(self, rows: Sequence[LayerCache]) -> "LayerCache":
        convs = uniform(rows, ConvCache)
        for conv in convs:
            if isinstance(conv, DeltaCache):
                raise mixed(conv)
        return BatchedConvCache(convs)

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
    def is_fixable(self) -> bool:
        return self.window is not None and self.state is not None

    def fixed(self, capacity: int) -> "LayerCache":
        return FixedDeltaCache.promote(self)

    def batched(self, rows: Sequence[LayerCache]) -> "LayerCache":
        return BatchedDeltaCache(uniform(rows, DeltaCache))

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


class FixedConvCache(ConvCache):
    """A `ConvCache` whose window lives in a graph-visible container.

    The growing cache holds it as a plain attribute, which `mx.compile` cannot see: an
    attribute rebound inside a traced function is rebound at trace time and never again.
    Here it sits in `graph`, the list a compiled decode passes as `inputs`/`outputs`, and
    the mixer's reads and writes go through a property over its slot — so the conv layer
    runs unchanged over either cache.
    """

    def __init__(self, window: mx.array, position: int) -> None:
        LayerCache.__init__(self)
        self.offset = position
        self.graph = [window]

    @classmethod
    def promote(cls, cache: ConvCache) -> "FixedConvCache":
        """Copy a completed prefill cache into the compile-friendly form. The window is set
        by then: a prefill that never ran has nothing worth compiling over."""
        window = cache.window
        if window is None:
            raise ValueError("promoting a conv cache before its prefill filled it")
        return cls(window, cache.offset)

    @property
    def is_fixable(self) -> bool:
        return True

    def fixed(self, capacity: int) -> "LayerCache":
        return self

    @property
    def containers(self) -> list[mx.array] | None:
        return self.graph

    @property
    def window(self) -> mx.array | None:
        return self.graph[0]

    @window.setter
    def window(self, value: mx.array | None) -> None:
        assert value is not None, "a fixed conv cache never clears its window"
        self.graph[0] = value


class FixedDeltaCache(FixedConvCache, DeltaCache):
    """`FixedConvCache` plus the DeltaNet's recurrent state, in the same container.

    With `checkpoints` taken, the container is five long and the last three are the round's:
    the state after every row, the raw conv-input rows of the round, and the window as it
    stood before it. They are what a rejected proposal is undone from — a recurrence keeps
    no history to subtract, so the only way back to row `kept` is to have kept row `kept`.
    """

    _CHECKPOINTS = 5

    def __init__(self, window: mx.array, state: mx.array, position: int) -> None:
        LayerCache.__init__(self)
        self.offset = position
        self.graph = [window, state]

    @property
    def checkpointing(self) -> bool:
        """Whether the round's slots are here to be filled. The mixer reads it to choose
        between the walk that records every row and the one that keeps only the last."""
        return len(self.graph) == self._CHECKPOINTS

    def checkpoints(self, rows: int) -> bool:
        """`LayerCache.checkpoints`. Three slots sized to the round, once — a second call at
        the same width finds them already there, which is what a regrow does."""
        window, state = self.graph[0], self.graph[1]
        if self.checkpointing and self.graph[2].shape[0] == rows:
            return True
        del self.graph[2:]
        self.graph.extend(
            [
                mx.zeros((rows, *state.shape[1:]), dtype=state.dtype),
                mx.zeros((rows, window.shape[-1]), dtype=window.dtype),
                mx.zeros_like(window),
            ]
        )
        return True

    def rewind(self, kept: int, position: int) -> None:
        """`LayerCache.rewind`. A slot pick and a slice, out of what the round recorded.

        The window is rebuilt from the one this round started with plus the conv rows that
        survived, cut to the `kernel - 1` the mixer reads — which is the window's own row
        count, so nothing here needs a config. The state is the one recorded after row
        `kept - 1`, and a round always keeps at least the gap it was fed, so that index
        exists.
        """
        assert self.checkpointing, "a rewind reads the slots `checkpoints` makes room for"
        tail = self.graph[0].shape[1]
        rebuilt = mx.concatenate([self.graph[4][0], self.graph[3][:kept]], axis=0)
        self.graph[0] = rebuilt[None, -tail:]
        self.graph[1] = self.graph[2][kept - 1][None]
        self.offset = position

    @classmethod
    def promote(cls, cache: ConvCache) -> "FixedDeltaCache":
        """Copy a completed prefill cache into the compile-friendly form. Both tensors are
        set by then: a prefill that never ran has nothing worth compiling over."""
        if not isinstance(cache, DeltaCache):
            raise TypeError(
                f"a fixed delta cache promotes a delta cache, not {type(cache).__name__}"
            )
        window, state = cache.window, cache.state
        if window is None or state is None:
            raise ValueError("promoting a delta cache before its prefill filled it")
        return cls(window, state, cache.offset)

    @property
    def state(self) -> mx.array | None:
        return self.graph[1]

    @state.setter
    def state(self, value: mx.array | None) -> None:
        assert value is not None, "a fixed delta cache never clears its state"
        self.graph[1] = value


class BatchedConvCache(LayerCache):
    """N `ConvCache`s as one: their windows stacked along the batch axis.

    A recurrent mixer is not read through `attend`; it reads `window` and writes it back,
    and each row's window already carries a leading axis of 1. So the getter concatenates
    and the setter splits, reassigning per row — which is what `ConvCache` itself does, and
    what makes the split exact rather than a view into a shared buffer.
    """

    def __init__(self, caches: Sequence[ConvCache]) -> None:
        self._caches = tuple(caches)

    @property
    def sequences(self) -> tuple[ConvCache, ...]:
        return self._caches

    @property
    def offset(self) -> int:
        return self._caches[0].offset

    @offset.setter
    def offset(self, value: int) -> None:
        # The mixer writes `offset + length`; every row advanced by the same step, so what
        # propagates is the delta and not the value — the rows are ragged.
        advance = value - self._caches[0].offset
        for cache in self._caches:
            cache.offset += advance

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        return sum(cache.nbytes for cache in self._caches)

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        return tuple(tensor for cache in self._caches for tensor in cache.tensors)

    @property
    def window(self) -> mx.array | None:
        return _stack([cache.window for cache in self._caches])

    @window.setter
    def window(self, value: mx.array | None) -> None:
        for index, cache in enumerate(self._caches):
            cache.window = None if value is None else value[index : index + 1]


class BatchedDeltaCache(BatchedConvCache):
    """`BatchedConvCache` plus the DeltaNet's recurrent state, stacked the same way."""

    def __init__(self, caches: Sequence[DeltaCache]) -> None:
        super().__init__(caches)
        self._delta = tuple(caches)

    @property
    def state(self) -> mx.array | None:
        return _stack([cache.state for cache in self._delta])

    @state.setter
    def state(self, value: mx.array | None) -> None:
        for index, cache in enumerate(self._delta):
            cache.state = None if value is None else value[index : index + 1]


def _stack(parts: Sequence[mx.array | None]) -> mx.array | None:
    """The rows as one batch, or nothing if no row has been filled yet. A mix of the two is
    a batch whose rows disagree about whether they were prefilled, which no mixer can read."""
    filled = [part for part in parts if part is not None]
    if not filled:
        return None
    if len(filled) != len(parts):
        raise ValueError("a recurrent batch mixes prefilled and empty rows")
    return mx.concatenate(filled)
