"""One compiled decode for every family: promotion, regrow+recompile, epoch staleness,
validity masks, and graph residency — written three times today, once here."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx

from mlx_omnia.engine.core.cache import (
    FixedDeltaCache,
    FixedKVCache,
    KVCache,
    LayerCache,
    RingKVCache,
    fit,
)

type StepFn = Callable[[mx.array, Sequence[LayerCache], mx.array | None], mx.array]
"""step(ids, slots, mask) -> logits.

- ids: what the stream hands the decode; the family decides the reshape.
- slots: the promoted caches (fixed shapes), one per layer.
- mask: the validity mask `columns <= anchor.position`, shape [1, 1, 1, capacity], built
  inside the traced forward so it reads the anchor's live position; None when the trunk
  has no full-attention layer.
- returns logits [1, vocab], last row already selected.

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


def state_of(slots: Sequence[LayerCache]) -> list[list[mx.array]]:
    """The containers the graph sees. Layers holding no container stay out."""
    state: list[list[mx.array]] = []
    for layer in slots:
        if isinstance(layer, FixedKVCache):
            state.append(layer.state)
        elif isinstance(layer, FixedDeltaCache):
            state.append(layer.graph)
        elif isinstance(layer, RingKVCache) and layer.state is not None:
            state.append(layer.state)
    return state


def anchor_of(slots: Sequence[LayerCache]) -> FixedKVCache | None:
    """The first `FixedKVCache`, owner of the position the mask reads."""
    return next((layer for layer in slots if isinstance(layer, FixedKVCache)), None)


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
        anchor = anchor_of(slots)
        columns = None if anchor is None else mx.arange(fitting)
        plan.prepare(slots)

        def forward(ids: mx.array) -> mx.array:
            # Read before any layer advances it: the row this step writes lands at the
            # pre-update position, so `<=` keeps it attendable and `<` would drop it.
            mask = (
                None
                if anchor is None or columns is None
                else (columns <= anchor.position).reshape(1, 1, 1, fitting)
            )
            return plan.step(ids, slots, mask)

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
