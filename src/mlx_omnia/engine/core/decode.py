"""One compiled decode for every family: promotion, regrow+recompile, epoch staleness and
graph residency — written three times today, once here."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx

from mlx_omnia.engine.core.api import Draftable, LanguageModel, Screened, Tracing
from mlx_omnia.engine.core.cache import (
    Composite,
    FixedKVCache,
    KVCache,
    LayerCache,
    RingKVCache,
    fit,
)

type StepFn = Callable[[mx.array, Sequence[LayerCache]], mx.array]
"""step(ids, slots) -> logits.

- ids: what the stream hands the decode; the family decides the reshape.
- slots: the promoted caches (fixed shapes), one per layer.
- returns logits [1, vocab], last row already selected.

No mask. A promoted buffer hands back its whole capacity and cuts the columns it has not
written itself (`LayerCache.readable`), which is the only form that also survives a ragged
batch — so there is nothing left for the caller to build or thread through.

Must be traceable by `mx.compile`. Weights are deliberately NOT captured: `mx.compile`
swaps arrays inside captured containers for tracers in place, and a frozen strategy field
reads its original object — which trips the uncaptured-input check the moment that object
is also in the capture list. Left out, they bake into the trace as constants, which is what
a decode closure wants; only what changes across steps enters the containers.
"""


def _noop(slots: Sequence[LayerCache]) -> None: ...


def _no_epochs() -> tuple[int, ...]:
    return ()


def _no_captures() -> list[object]:
    return []


@dataclass(frozen=True)
class DecodePlan:
    step: StepFn
    promote: Callable[[list[LayerCache], int], list[LayerCache]]
    """Growing caches -> fixed shapes at the given capacity. Must be idempotent over layers
    already fixed, and regrow a `FixedKVCache` sitting below the capacity. The returned list
    becomes the new slots; the caller assigns it back into the cache list."""
    prepare: Callable[[Sequence[LayerCache]], None] = _noop
    """Per-build hook: kernels, atlases, banks, resident wiring."""
    epochs: Callable[[], tuple[int, ...]] = _no_epochs
    """Staleness token. A change rebuilds in the same room."""
    captures: Callable[[], list[object]] = _no_captures
    """Extra live containers for the compile's inputs/outputs."""


def plan_of(model: LanguageModel[LayerCache], *, screened: bool = False) -> DecodePlan:
    """The plan the engine writes for any trunk, from what the trunk already declares.

    `step` is the forward and `promote` is `LayerCache.fixed`; neither varies by
    architecture, which is why neither is a family's to supply. What a family may still
    answer is `core.api.Tracing` — the resolution a trace cannot do and the containers it
    must be handed — and this reads it if it is there.

    Parameters
    ----------
    model : LanguageModel[LayerCache]
        The trunk. Measured at 1.025x laguna's own hand-written plan, which is what
        retired it.
    screened : bool, optional
        Take `core.api.Screened.screened` instead of the forward. The caller is asserting
        that nothing but `argmax` will read the result; `Screened` documents what that
        costs to get wrong. Ignored by a trunk that does not implement it.
    """
    step_fn = (
        model.screened if screened and isinstance(model, Screened) else model.__call__
    )

    def promote(current: list[LayerCache], fitting: int) -> list[LayerCache]:
        promoted = [layer.fixed(fitting) for layer in current]
        mx.eval(*(tensor for layer in promoted for tensor in layer.tensors))
        return promoted

    def step(ids: mx.array, slots: Sequence[LayerCache]) -> mx.array:
        return step_fn(ids[None], slots)[:, -1, :]

    if not isinstance(model, Tracing):
        return DecodePlan(step=step, promote=promote)

    captured: list[object] = []

    def prepare(slots: Sequence[LayerCache]) -> None:
        captured[:] = model.before_trace(slots)

    return DecodePlan(
        step=step,
        promote=promote,
        prepare=prepare,
        epochs=lambda: model.trace_epoch,
        captures=lambda: [*captured],
    )


def compiled_verify(
    model: Draftable[LayerCache],
    cache: list[LayerCache],
    *,
    rows: int,
    taps: Sequence[int],
) -> tuple[Callable[[mx.array], tuple[mx.array, mx.array]], Callable[[int, int], None]] | None:
    """A speculative round's fixed-shape half: one graph over the `rows` a round always
    feeds, and a rewind that costs no forward.

    Every round verifies exactly `width + 1` positions — the one committed token the target's
    cache has not seen, plus the proposals — so the shape never moves and the graph is built
    once. That is the same trade `compiled_decode` makes one row at a time, and the same
    pieces make it: `LayerCache.fixed` for the shapes, `LayerCache.checkpoints` for the room
    a rejection needs, `core.api.Tracing` for what a trace cannot settle for itself, and
    `Draftable.block_outputs` for the forward — which is the trunk's ordinary one, because a
    promoted buffer masks its own columns and the band that cuts them is also the causal one
    over these rows.

    `None` is a target the round rewinds by restoring and replaying instead: one whose
    layers cannot present a fixed shape, or one whose mixers do not record the state after
    every row (`Draftable.checkpoints`).
    """
    if not cache or not all(layer.is_fixable for layer in cache):
        return None
    if not model.checkpoints(cache, rows):
        return None

    def build(
        fitting: int,
    ) -> tuple[Callable[[mx.array], tuple[mx.array, mx.array]], int, tuple[int, ...]]:
        # A generation that outgrows the fixed attention buffers promotes again into larger
        # ones and recompiles; the recurrent containers are size-free and stay.
        slots = [layer.fixed(fitting) for layer in cache]
        granted = [layer.checkpoints(rows) for layer in slots]
        assert all(granted), "this target claimed a rewind its promoted layers cannot hold"
        mx.eval(*(tensor for layer in slots for tensor in layer.tensors))
        cache[:] = slots
        if isinstance(model, Tracing):
            model.before_trace(slots)
        state = state_of(slots)

        def forward(ids: mx.array) -> tuple[mx.array, mx.array]:
            logits, features = model.block_outputs(ids[None], slots, at=taps)
            return logits[0], features

        compiled = mx.compile(forward, inputs=state, outputs=state)
        return compiled, fitting, model.trace_epoch if isinstance(model, Tracing) else ()

    offset = cache[0].offset
    compiled, room, epochs = build(fit(offset))
    current = offset
    before = offset

    def verify(ids: mx.array) -> tuple[mx.array, mx.array]:
        nonlocal compiled, room, epochs, current, before
        if current + rows > room:
            compiled, room, epochs = build(fit(current))
        elif isinstance(model, Tracing) and model.trace_epoch != epochs:
            compiled, room, epochs = build(room)
        before = current
        logits, features = compiled(ids)
        current = before + rows
        for layer in cache:
            layer.offset = current
        return logits, features

    def rewind(kept: int, position: int) -> None:
        nonlocal current
        current = position
        for layer in cache:
            layer.rewind(kept, position)

    return verify, rewind


def state_of(slots: Sequence[LayerCache]) -> list[list[mx.array]]:
    """The containers the graph sees. Layers holding no container stay out; a `Composite`
    with no shared list of its own answers through its parts — each part keeps rebinding
    its own list, so they enter separately rather than flattened into one."""
    state: list[list[mx.array]] = []
    for layer in slots:
        held = layer.containers
        if held is not None:
            state.append(held)
        elif isinstance(layer, Composite):
            state.extend(state_of(list(layer.parts.values())))
    return state


def compiled_decode(
    plan: DecodePlan, cache: list[LayerCache], capacity: int | None = None
) -> Callable[[mx.array], mx.array]:
    """Promote a completed prefill cache and compile one-token forwards."""

    def build(
        fitting: int,
    ) -> tuple[Callable[[mx.array], mx.array], list[LayerCache], int, tuple[int, ...]]:
        slots = plan.promote(cache, fitting)
        cache[:] = slots
        state = state_of(slots)
        plan.prepare(slots)

        def forward(ids: mx.array) -> mx.array:
            return plan.step(ids, slots)

        inputs: list[object] = [*plan.captures(), state]
        compiled = mx.compile(forward, inputs=inputs, outputs=state)
        return compiled, slots, fitting, plan.epochs()

    offset = cache[0].offset
    room = capacity if capacity is not None else fit(offset)
    if offset >= room:
        room = fit(offset)
    compiled, slots, room, epochs = build(room)
    base = offset
    steps = 0

    def decode(ids: mx.array) -> mx.array:
        # The python-side offsets are assigned, not incremented: the trace's own pass
        # through the layers already bumped them once, and only once.
        nonlocal compiled, slots, room, epochs, steps
        if base + steps + 1 >= room:
            compiled, slots, room, epochs = build(fit(base + steps))
        elif plan.epochs() != epochs:
            compiled, slots, room, epochs = build(room)
        logits = compiled(ids)
        steps += 1
        for layer in slots:
            layer.offset = base + steps
        return logits

    return decode


class SlotBucket(NamedTuple):
    slots: tuple[tuple[LayerCache, ...], ...]
    decode: Callable[[mx.array], mx.array]


class Buckets:
    """Compiled graphs kept resident per (mode, batch, capacity), with slots moving in and
    out. A hit reuses the graph and reloads the slots when the identities diverge."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, int, int], SlotBucket] = {}

    def lease(
        self,
        mode: str,
        caches: Sequence[list[LayerCache]],
        capacity: int,
        build: Callable[[Sequence[list[LayerCache]], int], SlotBucket],
    ) -> SlotBucket:
        key = (mode, len(caches), capacity)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = build(caches, capacity)
            self._buckets[key] = bucket
        else:
            load_slots(caches, bucket.slots)
        return bucket

    def clear(self) -> None:
        self._buckets.clear()


def load_slots(
    caches: Sequence[list[LayerCache]],
    slots: Sequence[Sequence[LayerCache]],
) -> None:
    """Move live cache rows into the resident slots a compiled graph already holds.

    A no-op when every identity already matches. Otherwise the rows are snapshotted first —
    a source may be a slot another sequence is about to be written into — and then restored
    per cache class."""
    if all(
        all(source is target for source, target in zip(sequence, slot, strict=True))
        for sequence, slot in zip(caches, slots, strict=True)
    ):
        return
    snapshots: list[tuple[int, mx.array, mx.array]] = []
    for sequence, slot in zip(caches, slots, strict=True):
        for source, target in zip(sequence, slot, strict=True):
            if isinstance(target, RingKVCache) and isinstance(source, KVCache):
                source = RingKVCache.promote(source, target.window)
            assert isinstance(source, KVCache | FixedKVCache | RingKVCache)
            keys, values = source.fetch()
            snapshots.append(
                (source.rows, keys + mx.zeros_like(keys), values + mx.zeros_like(values))
            )
    mx.eval(*(array for _, keys, values in snapshots for array in (keys, values)))

    index = 0
    for sequence, slot in zip(caches, slots, strict=True):
        for target in slot:
            rows, keys, values = snapshots[index]
            index += 1
            if isinstance(target, FixedKVCache):
                target.restore(rows, {"keys": keys[..., :rows, :], "values": values[..., :rows, :]})
            elif isinstance(target, RingKVCache):
                target.reload(rows, keys, values)
            else:
                raise TypeError(f"compiled slot cannot use {type(target).__name__}")
        sequence[:] = slot
    mx.eval(*(tensor for slot in slots for layer in slot for tensor in layer.tensors))
