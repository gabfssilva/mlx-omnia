"""Continuous batching primitives."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import mlx.core as mx

from mlx_omnia.engine.core.attend import AttentionMask, KVStore
from mlx_omnia.engine.core.cache import (
    ConvCache,
    DeltaCache,
    FixedKVCache,
    KVCache,
    LayerCache,
    RingKVCache,
    fit,
    regrow,
)
from mlx_omnia.engine.core.decode import Buckets, SlotBucket
from mlx_omnia.engine.core.prefill import BLOCK
from mlx_omnia.engine.generate import (
    BlockOutputs,
    Constraint,
    ConstraintConflict,
    Meter,
    Penalty,
    ReasoningBudget,
    ReasoningWalk,
    Sampler,
    greedy,
)
from mlx_omnia.engine.speculative import (
    Proposer,
    SpeculationRefused,
    Verifiable,
    _rewinds,
    _round,
)

__all__ = [
    "BatchPrefill",
    "BatchSequence",
    "BatchedConvCache",
    "BatchedDeltaCache",
    "BatchedKVCache",
    "BatchedLayer",
    "batch",
    "prepare_batch_sequence",
    "step",
]


type BatchedLayer = KVStore | BatchedDeltaCache | BatchedConvCache
"""What a layer of a ragged batch can be: an attention adapter, or one of the recurrent
adapters a hybrid's mixer reads through `window`/`state`."""


@runtime_checkable
class BatchModel(Protocol):
    continuous_batching: bool

    def make_cache(self) -> list[KVCache]: ...

    def __call__(self, ids: mx.array, cache: Sequence[BatchedLayer]) -> mx.array: ...


@runtime_checkable
class BatchDecoder(Protocol):
    def batch_decode(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> mx.array: ...


@runtime_checkable
class BatchGreedyDecoder(Protocol):
    def batch_greedy(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> tuple[mx.array, ...]: ...


@runtime_checkable
class SingleDecoder(Protocol):
    def single_decode(
        self,
        ids: mx.array,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        *,
        capacity: int,
    ) -> mx.array: ...


@runtime_checkable
class SingleGreedyDecoder(Protocol):
    def single_greedy(
        self,
        ids: mx.array,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        *,
        capacity: int,
    ) -> mx.array: ...


@runtime_checkable
class PreparedSingleGreedyDecoder(Protocol):
    def prepare_single_greedy(
        self,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        *,
        capacity: int,
    ) -> Callable[[mx.array], mx.array]: ...


@dataclass
class BatchSequence:
    """Mutable decode state owned by one scheduler slot."""

    cache: list[KVCache | FixedKVCache | RingKVCache]
    pending: mx.array
    sampler: Sampler
    stop: Collection[int]
    remaining: int
    history: mx.array
    tokens: list[int]
    capacity: int
    penalty: Penalty | None = None
    constraint: Constraint | None = None
    meter: Meter | None = None
    finished: bool = False
    single_greedy: Callable[[mx.array], mx.array] | None = None
    single_slots: tuple[object, ...] = ()
    budget: ReasoningWalk | None = None
    owed: list[int] = field(default_factory=list)
    """The closer, between the step the budget armed on and the steps that feed it out."""
    proposer: Proposer | None = None
    """What writes this slot's draft, when the request could be verified greedily. A slot
    that holds one runs its own round inside the tick instead of joining the shared forward:
    its cache rewinds, which the compiled buckets' fixed buffers cannot.

    `tokens` is the history the round reads — prompt plus everything committed — and
    `pending` is the one id past it the cache has not seen, which is the round's gap."""


class BatchedKVCache:
    """Present independent sequence caches as one ragged attention batch.

    Parameters
    ----------
    caches : Sequence[KVCache]
        One cache for each row of the model batch.
    """

    def __init__(self, caches: Sequence[KVCache | FixedKVCache | RingKVCache]) -> None:
        self._caches = tuple(caches)

    @property
    def offset(self) -> mx.array:
        positions = [
            cache.position
            if isinstance(cache, (FixedKVCache, RingKVCache))
            else mx.array([cache.offset], dtype=mx.int32)
            for cache in self._caches
        ]
        return mx.concatenate(positions)

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
    ) -> mx.array:
        attended: list[mx.array] = []
        for index, cache in enumerate(self._caches):
            history_keys, history_values = cache.update_and_fetch(
                keys[index : index + 1], values[index : index + 1]
            )
            effective: AttentionMask = None if isinstance(mask, str) else mask
            if isinstance(effective, mx.array):
                if effective.shape[0] == len(self._caches):
                    effective = effective[index : index + 1]
                effective = effective[..., : history_keys.shape[2]]
            if isinstance(cache, (FixedKVCache, RingKVCache)):
                valid = mx.arange(history_keys.shape[2]) < cache.position
                valid = valid.reshape(1, 1, 1, -1)
                effective = (
                    valid
                    if isinstance(cache, RingKVCache) or effective is None
                    else valid & effective
                )
            attended.append(
                mx.fast.scaled_dot_product_attention(
                    queries[index : index + 1],
                    history_keys,
                    history_values,
                    scale=scale,
                    mask=effective,
                    sinks=sinks,
                )
            )
        return mx.concatenate(attended)


class BatchedConvCache:
    """N `ConvCache`s as one: their windows stacked along the batch axis.

    A recurrent mixer is not read through `attend`; it reads `window` and writes it back,
    and each row's window already carries a leading axis of 1. So the getter concatenates
    and the setter splits, reassigning per row — which is what `ConvCache` itself does, and
    what makes the split exact rather than a view into a shared buffer.
    """

    def __init__(self, caches: Sequence[ConvCache]) -> None:
        self._caches = tuple(caches)

    @property
    def rows(self) -> int:
        return len(self._caches)

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


def batch(caches: Sequence[Sequence[LayerCache]]) -> list[BatchedLayer]:
    """Transpose per-sequence caches into per-layer ragged batch adapters.

    Parameters
    ----------
    caches : Sequence[Sequence[LayerCache]]
        Cache layers grouped by sequence.

    Returns
    -------
    list[BatchedLayer]
        Cache adapters grouped by model layer, each one the adapter its layer's kind of
        cache is read through.
    """
    return [_adapt(layers) for layers in zip(*caches, strict=True)]


def _adapt(layers: tuple[LayerCache, ...]) -> BatchedLayer:
    """The adapter this layer's kind of cache is read through — the kind of the first row,
    with every other row required to match it."""
    head = layers[0]
    if isinstance(head, (KVCache, FixedKVCache, RingKVCache)):
        stores: list[KVCache | FixedKVCache | RingKVCache] = []
        for layer in layers:
            if not isinstance(layer, (KVCache, FixedKVCache, RingKVCache)):
                raise TypeError(_MIXED.format(kind=type(layer).__name__))
            stores.append(layer)
        return BatchedKVCache(stores)
    if isinstance(head, DeltaCache):
        deltas: list[DeltaCache] = []
        for layer in layers:
            if not isinstance(layer, DeltaCache):
                raise TypeError(_MIXED.format(kind=type(layer).__name__))
            deltas.append(layer)
        return BatchedDeltaCache(deltas)
    if isinstance(head, ConvCache):
        convs: list[ConvCache] = []
        for layer in layers:
            if not isinstance(layer, ConvCache) or isinstance(layer, DeltaCache):
                raise TypeError(_MIXED.format(kind=type(layer).__name__))
            convs.append(layer)
        return BatchedConvCache(convs)
    raise TypeError(f"{type(head).__name__} has no ragged batch adapter")


_MIXED = "a batched layer mixes {kind} with another cache kind"


type SequenceCache = list[KVCache | FixedKVCache | RingKVCache]

_BUCKET_SIZES = (1, 2, 4, 8)
_GENERIC = "generic"
_GENERIC_STATE = "_omnia_generic_batch"


@dataclass
class _GenericState:
    """Per-model residency for the generic compiled bucket. It hangs off the model instance
    rather than a weak table because `nn.Module` subclasses `dict` and is unhashable."""

    buckets: Buckets = field(default_factory=Buckets)
    dead: dict[tuple[int, int, int], SequenceCache] = field(default_factory=dict)


def _generic_state(model: BatchModel) -> _GenericState:
    existing = getattr(model, _GENERIC_STATE, None)
    if isinstance(existing, _GenericState):
        return existing
    fresh = _GenericState()
    object.__setattr__(model, _GENERIC_STATE, fresh)
    return fresh


def _kv_pure(caches: Sequence[Sequence[LayerCache]]) -> bool:
    """Whether every layer of every row is a plain growing or fixed KV buffer.

    Strict on the class: a `QuantizedKVCache`, a ring, a shared reader or a recurrent state
    is a different arithmetic, and the generic graph only reproduces this one. Rows are
    required too: a layer no prefill ever wrote has nothing to promote, and a model that
    never touches its cache (a test double) must stay on the eager path it was written
    against."""
    return all(
        (type(layer) is KVCache and layer.offset > 0 and layer.tensors)
        or type(layer) is FixedKVCache
        for sequence in caches
        for layer in sequence
    )


def _promote_row(sequence: Sequence[LayerCache], capacity: int) -> list[FixedKVCache]:
    slot: list[FixedKVCache] = []
    for layer in sequence:
        if isinstance(layer, FixedKVCache):
            slot.append(layer if layer.state[0].shape[2] == capacity else regrow(layer, capacity))
        elif isinstance(layer, KVCache):
            slot.append(FixedKVCache.promote(layer, capacity))
        else:
            raise TypeError(f"generic compiled batch cannot use {type(layer).__name__}")
    return slot


def _build_generic(
    model: BatchModel, caches: Sequence[SequenceCache], capacity: int
) -> SlotBucket:
    slots = [_promote_row(sequence, capacity) for sequence in caches]
    mx.eval(*(tensor for slot in slots for layer in slot for tensor in layer.tensors))
    stores = batch(slots)
    state = [layer.state for slot in slots for layer in slot]

    def forward(ids: mx.array) -> mx.array:
        return model(ids, stores)[:, -1, :]

    compiled = mx.compile(forward, inputs=state, outputs=state)
    for sequence, slot in zip(caches, slots, strict=True):
        sequence[:] = slot
    return SlotBucket(tuple(tuple(slot) for slot in slots), compiled)


def _dead_row(
    model: BatchModel, state: _GenericState, size: int, capacity: int, row: int
) -> SequenceCache:
    """A bucket row no sequence occupies, kept resident so the padding never reloads. One
    token of its own gives every layer a position, and `BatchedKVCache` masks the row to it,
    so it attends its own stale state and the caller drops the result."""
    key = (size, capacity, row)
    dead = state.dead.get(key)
    if dead is None:
        fresh = model.make_cache()
        model(mx.array([[0]]), fresh)
        mx.eval(*(tensor for layer in fresh for tensor in layer.tensors))
        dead = [*fresh]
        state.dead[key] = dead
    return dead


def _generic_bucket(
    model: BatchModel, caches: Sequence[SequenceCache], capacity: int
) -> SlotBucket | None:
    """The compiled bucket for a batch whose every layer is a plain KV cache, or None when
    some layer is not — the caller falls back to the eager ragged forward."""
    count = len(caches)
    if count > _BUCKET_SIZES[-1] or not _kv_pure(caches):
        return None
    state = _generic_state(model)
    size = next(candidate for candidate in _BUCKET_SIZES if count <= candidate)
    padded: list[SequenceCache] = [
        *caches,
        *(_dead_row(model, state, size, capacity, row) for row in range(count, size)),
    ]
    if not _kv_pure(padded):
        return None
    views: list[list[LayerCache]] = [[*sequence] for sequence in padded]

    def build(_pending: Sequence[list[LayerCache]], room: int) -> SlotBucket:
        return _build_generic(model, padded, room)

    bucket = state.buckets.lease(_GENERIC, views, capacity, build)
    for sequence, slot in zip(padded, bucket.slots, strict=True):
        sequence[:] = _stores(slot)
    return bucket


def _stores(slot: Sequence[LayerCache]) -> SequenceCache:
    narrowed: SequenceCache = []
    for layer in slot:
        if not isinstance(layer, (KVCache, FixedKVCache, RingKVCache)):
            raise TypeError(f"generic compiled slot cannot use {type(layer).__name__}")
        narrowed.append(layer)
    return narrowed


def _pad_rows(ids: mx.array, size: int) -> mx.array:
    missing = size - ids.shape[0]
    if missing <= 0:
        return ids
    return mx.concatenate([ids, mx.repeat(ids[-1:], missing, axis=0)])


@dataclass
class BatchPrefill:
    """A prompt on its way to a `BatchSequence`, one block of prefill per `advance`.

    The split is `core.prefill.prefill` turned inside out: the same blocks, the same eval of
    the caches between them, the same whole last block carrying the head — but returning to
    the caller between blocks, so a joiner's prompt costs a block of the scheduler's tick
    instead of the whole tick.
    """

    model: BatchModel
    prompt: Sequence[int]
    cache: list[KVCache]
    at: int
    reused: int
    max_tokens: int
    sampler: Sampler
    stop: Collection[int] = ()
    penalty: Penalty | None = None
    constraint: Constraint | None = None
    meter: Meter | None = None
    reasoning: ReasoningBudget | None = None
    proposer: Proposer | None = None
    block: int = BLOCK
    history: mx.array = field(init=False)
    _remaining: mx.array = field(init=False)

    def __post_init__(self) -> None:
        if self.reasoning is not None and self.constraint is not None:
            # The same refusal `stream_ids` raises, and for the same reason: the closer is fed
            # past the mask the matcher is advanced under.
            raise ConstraintConflict(
                "a reasoning budget and a grammar do not compose: the forced closer bypasses "
                "the mask the matcher is advanced under"
            )
        self.history = mx.array(self.prompt)
        self._remaining = self.history[None, self.reused :]

    def _feed(self, ids: mx.array) -> mx.array:
        """One block of prefill, plus what the proposer reads of the trunk when it reads
        anything: every row a block writes is kept, so the proposer is handed all of them."""
        proposer = self.proposer
        if proposer is None or not proposer.taps:
            return self.model(ids, self.cache)
        if not isinstance(self.model, BlockOutputs):
            raise SpeculationRefused(
                f"this draft conditions on the target's blocks {list(proposer.taps)} and the "
                "target has no `block_outputs`: what it would read instead is nothing at all"
            )
        logits, features = self.model.block_outputs(ids, self.cache, at=proposer.taps)
        proposer.absorb(features)
        return logits

    def advance(self) -> BatchSequence | None:
        """Feed one block; return the sequence once the last block has been read."""
        length = self._remaining.shape[1]
        if length - self.at > self.block:
            part = self._remaining[:, self.at : self.at + self.block]
            self._feed(part)
            mx.eval([tensor for layer in self.cache for tensor in layer.tensors])
            # Back to the system rather than into MLX's own pool, as `core.prefill` does.
            mx.clear_cache()
            self.at += self.block
            return None
        logits = self._feed(self._remaining[:, self.at : length])[:, -1, :]
        if self.penalty is not None:
            logits = self.penalty(logits, self.history)
        if self.constraint is not None:
            logits = self.constraint.mask(logits, self.max_tokens)
        pending = self.sampler(logits)[0]
        mx.async_eval(pending)
        if self.meter is not None:
            self.meter.prefill(len(self.prompt), self.reused)
        required = self.cache[0].offset + self.max_tokens
        capacity = (required + 255) // 256 * 256
        self.at = length
        return BatchSequence(
            self.cache,
            pending,
            self.sampler,
            self.stop,
            self.max_tokens,
            self.history,
            list(self.prompt),
            capacity,
            self.penalty,
            self.constraint,
            self.meter,
            finished=self.max_tokens <= 0,
            budget=None if self.reasoning is None else ReasoningWalk(self.reasoning),
            proposer=self.proposer,
        )


def prepare_batch_sequence(
    model: BatchModel,
    prompt: Sequence[int],
    *,
    max_tokens: int,
    sampler: Sampler,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    constraint: Constraint | None = None,
    meter: Meter | None = None,
    reasoning: ReasoningBudget | None = None,
    proposer: Proposer | None = None,
    cache: list[KVCache] | None = None,
    reused: int = 0,
) -> BatchSequence:
    """Prefill one prompt and return its first pending sampled token.

    Parameters
    ----------
    model : BatchModel
        Model that will later execute shared decode steps.
    prompt : Sequence[int]
        Tokenized prompt.
    max_tokens : int
        Maximum number of emitted tokens.
    sampler : Sampler
        Per-sequence sampling policy.
    stop : Collection[int], optional
        Token ids consumed but not emitted.
    penalty : Penalty | None, optional
        Per-sequence repetition penalty.
    constraint : Constraint | None, optional
        Per-sequence grammar state.
    meter : Meter | None, optional
        Request metrics sink.
    reasoning : ReasoningBudget | None, optional
        How many ids the reasoning block may spend, and the ids that close it.
    proposer : Proposer | None, optional
        What drafts for this sequence, when the request can be verified greedily.

    Returns
    -------
    BatchSequence
        State ready for the scheduler's first step.
    """
    prefill = BatchPrefill(
        model,
        prompt,
        model.make_cache() if cache is None else cache,
        0,
        reused,
        max_tokens,
        sampler,
        stop,
        penalty,
        constraint,
        meter,
        reasoning,
        proposer,
    )
    while (sequence := prefill.advance()) is None:
        pass
    return sequence


def step(model: BatchModel, sequences: Sequence[BatchSequence]) -> list[list[int]]:
    """Advance every active sequence through one shared model forward.

    Parameters
    ----------
    model : BatchModel
        Model shared by the active sequences.
    sequences : Sequence[BatchSequence]
        Active scheduler slots.

    Returns
    -------
    list[list[int]]
        The ids each slot emitted this tick; empty when its stop token was consumed.

    A slot that drafts is not part of the shared forward: its round feeds `width + 1` rows
    and rewinds whatever the target rejected, which no compiled bucket's fixed buffer can
    do. It runs its own round here and the rest of the batch steps as it always did.
    """
    drafting = [index for index, sequence in enumerate(sequences) if sequence.proposer is not None]
    if not drafting:
        return _shared_step(model, sequences)
    plain = [index for index, sequence in enumerate(sequences) if sequence.proposer is None]
    emitted: list[list[int]] = [[] for _ in sequences]
    advanced = _shared_step(model, [sequences[index] for index in plain]) if plain else []
    for index, ids in zip(plain, advanced, strict=True):
        emitted[index] = ids
    for index in drafting:
        emitted[index] = _speculative_tick(model, sequences[index])
    return emitted


def _speculative_tick(model: BatchModel, sequence: BatchSequence) -> list[int]:
    """One speculative round as a tick: the proposer writes, the target verifies the gap plus
    the draft in one forward, and what the target's own argmax confirmed is committed.

    The gap is exactly this slot's `pending` — the id drawn last tick that the cache has not
    seen — so the round's contract is the loop's: it hands back the accepted proposals plus
    the target's token past them, all of which continue `pending`. The ids emitted this tick
    are therefore `pending` and all but the last of them; the last becomes the new `pending`.
    """
    proposer = sequence.proposer
    assert proposer is not None, "only a drafting slot reaches the round"
    cache = sequence.cache
    replays = _rewinds(cache, "target")
    taps = proposer.taps

    def forward(ids: mx.array) -> tuple[mx.array, mx.array | None]:
        if not taps:
            return model(ids, cache), None
        if not isinstance(model, BlockOutputs):
            raise SpeculationRefused(
                f"this draft conditions on the target's blocks {list(taps)} and the target "
                "has no `block_outputs`: what it would read instead is nothing at all"
            )
        return model.block_outputs(ids, cache, at=taps)

    def verify(
        ids: mx.array,
    ) -> tuple[mx.array, mx.array | None, Callable[[int], None] | None]:
        if taps and isinstance(model, Verifiable):
            return model.verify(ids, cache, at=taps)
        logits, features = forward(ids)
        return logits, features, None

    gap = _int(sequence.pending)
    settled, accepted = _round(
        verify, forward, proposer, cache, [*sequence.tokens, gap], replays
    )
    if sequence.meter is not None:
        sequence.meter.speculation.rounds += 1
        sequence.meter.speculation.proposed += proposer.width
        sequence.meter.speculation.accepted += accepted
    sequence.pending = mx.array(settled[-1])
    emitted: list[int] = []
    for token in (gap, *settled[:-1]):
        sequence.tokens.append(token)
        if token in sequence.stop:
            sequence.finished = True
            break
        sequence.remaining -= 1
        if sequence.meter is not None:
            sequence.meter.token()
        emitted.append(token)
        if sequence.remaining == 0:
            sequence.finished = True
            break
    return emitted


def _int(value: mx.array) -> int:
    token = value.item()
    if not isinstance(token, int):
        raise TypeError(f"sampler returned {type(token).__name__}, expected int")
    return token


def _shared_step(model: BatchModel, sequences: Sequence[BatchSequence]) -> list[list[int]]:
    """Every slot that does not draft, through one forward."""
    count = len(sequences)
    if count == 1:
        sequence = sequences[0]
        if (
            sequence.single_greedy is not None
            and sequence.sampler is greedy
            and sequence.penalty is None
            and sequence.constraint is None
            and all(
                source is target
                for source, target in zip(
                    sequence.cache, sequence.single_slots, strict=True
                )
            )
        ):
            queued = sequence.single_greedy(sequence.pending[None])
            mx.async_eval(queued)
            return [_advance(sequence, sequence.pending, queued)]
    drawn_ids = tuple(sequence.pending for sequence in sequences)
    ids = drawn_ids[0][None, None] if count == 1 else mx.stack(drawn_ids)[:, None]
    caches = [sequence.cache for sequence in sequences]
    bucketed = 2 <= count <= 8
    unfiltered_greedy = all(
        sequence.sampler is greedy
        and sequence.penalty is None
        and sequence.constraint is None
        for sequence in sequences
    )
    prepared_single: Callable[[mx.array], mx.array] | None = None
    if count == 1 and unfiltered_greedy:
        sequence = sequences[0]
        if sequence.single_greedy is not None and all(
            source is target
            for source, target in zip(sequence.cache, sequence.single_slots, strict=True)
        ):
            prepared_single = sequence.single_greedy
        elif isinstance(model, PreparedSingleGreedyDecoder):
            prepared_single = model.prepare_single_greedy(
                sequence.cache, capacity=sequence.capacity
            )
            sequence.single_greedy = prepared_single
            sequence.single_slots = tuple(sequence.cache)
    elif count > 1:
        for sequence in sequences:
            sequence.single_greedy = None
            sequence.single_slots = ()
    single_greedy = (
        model
        if count == 1
        and unfiltered_greedy
        and prepared_single is None
        and isinstance(model, SingleGreedyDecoder)
        else None
    )
    single_decode = (
        model
        if count == 1
        and prepared_single is None
        and single_greedy is None
        and isinstance(model, SingleDecoder)
        else None
    )
    batch_greedy = (
        model
        if bucketed and unfiltered_greedy and isinstance(model, BatchGreedyDecoder)
        else None
    )
    batch_decode = (
        model
        if bucketed and batch_greedy is None and isinstance(model, BatchDecoder)
        else None
    )
    compiled = any(
        decoder is not None
        for decoder in (
            prepared_single,
            single_greedy,
            single_decode,
            batch_greedy,
            batch_decode,
        )
    )
    capacity = max(sequence.capacity for sequence in sequences) if compiled else 0
    generic: SlotBucket | None = None
    if not compiled:
        room = max(sequence.capacity for sequence in sequences)
        needed = max(sequence.cache[0].offset for sequence in sequences) + 1
        if needed >= room:
            room = fit(needed)
        generic = _generic_bucket(model, caches, room)
        if generic is not None:
            for sequence in sequences:
                sequence.capacity = room
    if prepared_single is not None:
        following = [prepared_single(ids[0])]
        logits = None
    elif single_greedy is not None:
        output = single_greedy.single_greedy(ids[0], caches[0], capacity=capacity)
        following = [output[index] for index in range(count)]
        logits = None
    elif batch_greedy is not None:
        output = batch_greedy.batch_greedy(ids, caches, capacity=capacity)
        following = [output[index] for index in range(count)]
        logits = None
    elif single_decode is not None:
        logits = single_decode.single_decode(ids[0], caches[0], capacity=capacity)
    elif batch_decode is not None:
        logits = batch_decode.batch_decode(ids, caches, capacity=capacity)[:, -1, :]
    elif generic is not None:
        logits = generic.decode(_pad_rows(ids, len(generic.slots)))[:count]
        # The graph moved each slot's position; the host-side counter is assigned back so the
        # next lease reads the live length without evaluating the tensor.
        for sequence in sequences:
            for layer in sequence.cache:
                layer.offset += 1
    else:
        logits = model(ids, batch(caches))[:, -1, :]
    walked: list[bool | None] = [None] * count
    if logits is not None:
        following = []
        for index, sequence in enumerate(sequences):
            row = logits[index : index + 1]
            if sequence.penalty is not None:
                sequence.history = mx.concatenate([sequence.history, sequence.pending[None]])
                row = sequence.penalty(row, sequence.history)
            if sequence.constraint is not None:
                # The mask of step n+1 is a function of the id step n drew, so the grammar
                # advances over the drawn id *before* masking — at the cost of the sync a
                # constrained request already pays (`stream_ids` documents the contract).
                # Masking first would draw n+1 from the state as of n-1, and the grammar
                # would be offered the same alternatives twice.
                token = _int(sequence.pending)
                walked[index] = (
                    None if token in sequence.stop else sequence.constraint.accept(token)
                )
                if walked[index]:
                    row = sequence.constraint.mask(row, sequence.remaining - 1)
            following.append(sequence.sampler(row)[0])
    mx.async_eval(*following)
    return [
        _advance(sequence, drawn, queued, accepted)
        for sequence, drawn, queued, accepted in zip(
            sequences, drawn_ids, following, walked, strict=True
        )
    ]


def _advance(
    sequence: BatchSequence,
    drawn: mx.array,
    queued: mx.array,
    accepted: bool | None = None,
) -> list[int]:
    """Commit this tick's id and decide what the next one draws from.

    An armed budget owes the closer, and the closer is fed rather than drawn: the queued step
    is spent — the cache owes `drawn` its row either way — and its result dropped, which is
    the whole cost of arming.
    """
    emitted = _commit(sequence, drawn, queued, accepted)
    if sequence.owed and not sequence.finished:
        sequence.pending = mx.array(sequence.owed.pop(0))
    return emitted


def _commit(
    sequence: BatchSequence,
    drawn: mx.array,
    queued: mx.array,
    accepted: bool | None = None,
) -> list[int]:
    """`accepted` is the grammar's verdict when the caller already advanced it over
    `drawn` (the shared step does, to mask the next draw from the right state); `None`
    asks this commit to advance it."""
    token = _int(drawn)
    sequence.pending = queued
    sequence.tokens.append(token)
    if token in sequence.stop:
        sequence.finished = True
        return []
    sequence.remaining -= 1
    if sequence.meter is not None:
        sequence.meter.token()
    if sequence.budget is not None:
        sequence.owed.extend(sequence.budget.emitted(token))
    if sequence.constraint is None:
        walked = True
    elif accepted is None:
        walked = sequence.constraint.accept(token)
    else:
        walked = accepted
    sequence.finished = sequence.remaining == 0 or not walked
    return [token]
