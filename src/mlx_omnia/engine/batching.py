"""Continuous batching primitives."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_omnia.engine.core.api import Draftable, LanguageModel, Proposer, Screened, Tracing
from mlx_omnia.engine.core.cache import (
    BatchedConvCache,
    BatchedDeltaCache,
    BatchedKVCache,
    BatchedLayerCache,
    BatchedSharedKVReader,
    LayerCache,
    batch,
    fit,
)
from mlx_omnia.engine.core.decode import Buckets, SlotBucket, compiled_decode, plan_of
from mlx_omnia.engine.core.prefill import BLOCK, landing
from mlx_omnia.engine.core.prefix import Prefix
from mlx_omnia.engine.generate import (
    BUDGET_WITH_GRAMMAR,
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
    one_round,
    rewinds,
    room,
    taps_refused,
)

__all__ = [
    "BatchPrefill",
    "BatchSequence",
    "BatchedConvCache",
    "BatchedDeltaCache",
    "BatchedKVCache",
    "BatchedLayerCache",
    "BatchedSharedKVReader",
    "batch",
    "prepare_batch_sequence",
    "step",
]


_BUCKET_SIZES = (1, 2, 4, 8)
_GENERIC = "generic"
_SCREENED = "screened"
_GENERIC_STATE = "_omnia_generic_batch"


type SequenceCache = list[LayerCache]
"""One sequence's layers. Any kind: what a layer can do about batching, about a fixed
shape and about spans is the layer's own answer, so nothing here has to enumerate them."""


@dataclass
class BatchSequence:
    """Mutable decode state owned by one scheduler slot."""

    cache: SequenceCache
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
    lease: Callable[[mx.array], mx.array] | None = None
    """This slot's resident one-row compiled decode, built once and kept while the slot runs
    alone. `lease_slots` is the cache it was compiled over: a bucket that promoted the rows
    out from under it leaves the graph writing somewhere the sequence no longer reads."""
    lease_slots: tuple[object, ...] = ()
    budget: ReasoningWalk | None = None
    owed: list[int] = field(default_factory=list)
    """The closer, between the step the budget armed on and the steps that feed it out."""
    proposer: Proposer | None = None
    """What writes this slot's draft, when the request could be verified greedily. A slot
    that holds one runs its own round inside the tick instead of joining the shared forward:
    its cache rewinds, which the compiled buckets' fixed buffers cannot.

    `tokens` is the history the round reads — prompt plus everything committed — and
    `pending` is the one id past it the cache has not seen, which is the round's gap."""
    prefix: Prefix | None = None
    """This sequence's walk along its conversation's chain, when the daemon keeps one. The
    decode closes a span on every boundary it crosses, which is the only moment it stands
    still; `tokens` is both the ids and the row count, because the forward feeds `pending`
    before the commit appends it."""


@dataclass
class _GenericState:
    """Per-model residency for the generic compiled bucket. It hangs off the model instance
    rather than a weak table because `nn.Module` subclasses `dict` and is unhashable."""

    buckets: Buckets = field(default_factory=Buckets)
    dead: dict[tuple[int, int, int], SequenceCache] = field(default_factory=dict)


def _generic_state(model: LanguageModel[LayerCache]) -> _GenericState:
    existing = getattr(model, _GENERIC_STATE, None)
    if isinstance(existing, _GenericState):
        return existing
    fresh = _GenericState()
    object.__setattr__(model, _GENERIC_STATE, fresh)
    return fresh


def _fixable(caches: Sequence[Sequence[LayerCache]]) -> bool:
    """Whether every layer of every row can present itself in a fixed shape.

    Asked of the layers and not of their classes: what a `QuantizedKVCache` or a family's
    pooled cache can do about a traced graph is its own answer, and one that says no leaves
    the batch on the eager ragged forward instead of being enumerated here."""
    return all(layer.is_fixable for sequence in caches for layer in sequence)


def _build_generic(
    model: LanguageModel[LayerCache],
    caches: Sequence[SequenceCache],
    capacity: int,
    *,
    screened: bool = False,
) -> SlotBucket:
    slots = [[layer.fixed(capacity) for layer in sequence] for sequence in caches]
    mx.eval(*(tensor for slot in slots for layer in slot for tensor in layer.tensors))
    stores = batch(slots)
    state = [held for slot in slots for layer in slot if (held := layer.containers) is not None]
    # The same hook the lease takes, for the same reason: a kernel resolved inside the trace
    # is resolved once and then frozen, and a resident table the step reads has to be an
    # input or the graph bakes in the array it saw. The adapters do the masking here, so
    # only the settling half of `Tracing` matters — which is why a trunk that declares it
    # gets a correct bucket whether or not it also gets the lease.
    captures = list(model.before_trace(stores)) if isinstance(model, Tracing) else []
    step = model.screened if screened and isinstance(model, Screened) else model.__call__

    def forward(ids: mx.array) -> mx.array:
        return step(ids, stores)[:, -1, :]

    compiled = mx.compile(forward, inputs=[*captures, state], outputs=state)
    for sequence, slot in zip(caches, slots, strict=True):
        sequence[:] = slot
    return SlotBucket(tuple(tuple(slot) for slot in slots), compiled)


def _dead_row(
    model: LanguageModel[LayerCache], state: _GenericState, size: int, capacity: int, row: int
) -> SequenceCache:
    """A bucket row no sequence occupies, kept resident so the padding never reloads. One
    token of its own gives every layer a position, and `BatchedKVCache` masks the row to it,
    so it attends its own stale state and the caller drops the result."""
    key = (size, capacity, row)
    dead = state.dead.get(key)
    if dead is None:
        fresh = list[LayerCache](model.make_cache())
        model(mx.array([[0]]), fresh)
        mx.eval(*(tensor for layer in fresh for tensor in layer.tensors))
        dead = fresh
        state.dead[key] = dead
    return dead


def _generic_bucket(
    model: LanguageModel[LayerCache],
    caches: Sequence[SequenceCache],
    capacity: int,
    *,
    screened: bool = False,
) -> SlotBucket | None:
    """The compiled bucket for a batch whose every layer can take a fixed shape, or None
    when some layer cannot — the caller falls back to the eager ragged forward."""
    count = len(caches)
    if count > _BUCKET_SIZES[-1] or not _fixable(caches):
        return None
    state = _generic_state(model)
    size = next(candidate for candidate in _BUCKET_SIZES if count <= candidate)
    padded: list[SequenceCache] = [
        *caches,
        *(_dead_row(model, state, size, capacity, row) for row in range(count, size)),
    ]
    if not _fixable(padded):
        return None
    views: list[list[LayerCache]] = [[*sequence] for sequence in padded]

    def build(_pending: Sequence[list[LayerCache]], room: int) -> SlotBucket:
        return _build_generic(model, padded, room, screened=screened)

    # The mode keys the residency: a screened graph and a logits graph over the same rows
    # are two graphs, and handing a request that samples the screened one would feed the
    # sampler numbers that were never computed.
    try:
        bucket = state.buckets.lease(_SCREENED if screened else _GENERIC, views, capacity, build)
    except NotImplementedError:
        # Fixable but not yet batchable: a layer whose fixed form has no ragged batch
        # adapter refuses inside `batch`, before anything is assigned back. The bucket
        # declines and the eager ragged forward serves the batch — the same fallback the
        # family had while it still declined `fixed`.
        return None
    for sequence, slot in zip(padded, bucket.slots, strict=True):
        sequence[:] = list(slot)
    return bucket


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

    model: LanguageModel[LayerCache]
    prompt: Sequence[int]
    cache: SequenceCache
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
    prefix: Prefix | None = None
    block: int = BLOCK
    history: mx.array = field(init=False)
    _remaining: mx.array = field(init=False)

    def __post_init__(self) -> None:
        if self.reasoning is not None and self.constraint is not None:
            # The same refusal `stream_ids` raises, and for the same reason: the closer is fed
            # past the mask the matcher is advanced under.
            raise ConstraintConflict(BUDGET_WITH_GRAMMAR)
        self.history = mx.array(self.prompt)
        self._remaining = self.history[None, self.reused :]

    def _feed(self, ids: mx.array) -> mx.array:
        """One block of prefill, plus what the proposer reads of the trunk when it reads
        anything: every row a block writes is kept, so the proposer is handed all of them."""
        proposer = self.proposer
        if proposer is None or not proposer.taps:
            return self.model(ids, self.cache)
        if not isinstance(self.model, Draftable):
            raise taps_refused(proposer.taps)
        logits, features = self.model.block_outputs(ids, self.cache, at=proposer.taps)
        proposer.absorb(features)
        return logits

    def advance(self) -> BatchSequence | None:
        """Feed one block; return the sequence once the last block has been read."""
        if self.meter is not None and self.meter.prefill_started is None:
            # On the *first* block and not the last. A 40k prompt is twenty blocks spread over
            # twenty scheduler ticks; a clock started at the last one reports the tail of the
            # prefill as the whole of it, which is a TTFT that cannot be compared with
            # anything — least of all with the same prompt resumed.
            self.meter.prefill(len(self.prompt), self.reused)
            self.meter.kept_prefix = self.prefix is not None
            if self.prefix is not None:
                self.meter.prefix = self.prefix.reuse
        length = self._remaining.shape[1]
        if length - self.at > self.block:
            part = self._remaining[:, self.at : self.at + self.block]
            self._feed(part)
            mx.eval([tensor for layer in self.cache for tensor in layer.tensors])
            self._stopped(self.at + self.block)
            # Back to the system rather than into MLX's own pool, as `core.prefill` does.
            mx.clear_cache()
            return None
        if self.prefix is not None and self.prefix.chain.anchored:
            # The last block cut on the last span boundary before the forward that reads the
            # logits, so the trunk stops on one. Without it a prompt whose end falls
            # mid-span — nearly every prompt, and every request asking for one token — leaves
            # a recurrent trunk no resume point at all, and the next turn prefills whole.
            cut = landing(self.reused + self.at, len(self.prompt), self.prefix.chain.span)
            if cut - self.reused > self.at:
                self._feed(self._remaining[:, self.at : cut - self.reused])
                mx.eval([tensor for layer in self.cache for tensor in layer.tensors])
                self._stopped(cut - self.reused)
        logits = self._feed(self._remaining[:, self.at : length])[:, -1, :]
        if self.penalty is not None:
            logits = self.penalty(logits, self.history)
        if self.constraint is not None:
            logits = self.constraint.mask(logits, self.max_tokens)
        pending = self.sampler(logits)[0]
        mx.async_eval(pending)
        required = self.cache[0].offset + self.max_tokens
        capacity = (required + 255) // 256 * 256
        self._stopped(length)
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
            prefix=self.prefix,
        )

    def _stopped(self, at: int) -> None:
        """The trunk now stands at `at` fresh rows: what the prefix store can close on, and
        what a reader watching a long prompt being read is told."""
        self.at = at
        if self.meter is not None:
            self.meter.fed(at)
        self._close()

    def _close(self) -> None:
        """The spans this prefill has finished, stored while the trunk is stopped."""
        if self.prefix is not None:
            self.prefix.commit(self.prompt, self.cache, self.reused + self.at)


def prepare_batch_sequence(
    model: LanguageModel[LayerCache],
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
    cache: SequenceCache | None = None,
    reused: int = 0,
) -> BatchSequence:
    """Prefill one prompt and return its first pending sampled token.

    Parameters
    ----------
    model : LanguageModel[LayerCache]
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
        list[LayerCache](model.make_cache()) if cache is None else cache,
        at=0,
        reused=reused,
        max_tokens=max_tokens,
        sampler=sampler,
        stop=stop,
        penalty=penalty,
        constraint=constraint,
        meter=meter,
        reasoning=reasoning,
        proposer=proposer,
    )
    while (sequence := prefill.advance()) is None:
        pass
    return sequence


def step(model: LanguageModel[LayerCache], sequences: Sequence[BatchSequence]) -> list[list[int]]:
    """Advance every active sequence through one shared model forward.

    Parameters
    ----------
    model : LanguageModel[LayerCache]
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


def _speculative_tick(model: LanguageModel[LayerCache], sequence: BatchSequence) -> list[int]:
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
    replays = rewinds(cache, "target")
    taps = proposer.taps

    def forward(ids: mx.array) -> tuple[mx.array, mx.array | None]:
        if not taps:
            return model(ids, cache), None
        if not isinstance(model, Draftable):
            raise taps_refused(taps)
        return model.block_outputs(ids, cache, at=taps)

    gap = _int(sequence.pending)
    settled, accepted = one_round(
        forward,
        proposer,
        cache,
        [*sequence.tokens, gap],
        replays,
        limit=room(sequence.prefix, len(sequence.tokens), proposer.width),
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
    _close(sequence)
    return emitted


def _int(value: mx.array) -> int:
    token = value.item()
    if not isinstance(token, int):
        raise TypeError(f"sampler returned {type(token).__name__}, expected int")
    return token


def _lease(
    model: LanguageModel[LayerCache],
    sequences: Sequence[BatchSequence],
    *,
    screened: bool,
) -> Callable[[mx.array], mx.array] | None:
    """One sequence's resident compiled decode, or `None` when this trunk cannot take one.

    The lease hands the trunk its promoted caches rather than a one-row ragged adapter,
    which is what keeps a fused single-row kernel on — measured at 1.147x the adapter on
    Laguna-XS. `core.api.Tracing` is the claim that the trunk survives it; the layers answer
    the other half through `is_fixable`.

    Kept on the sequence because building it compiles a graph. It is dropped the moment the
    cache it was compiled over is no longer the one the sequence holds — a tick spent in a
    bucket promotes the rows into the bucket's slots, and a graph still writing the old ones
    would decode from a history that stopped growing.
    """
    if not isinstance(model, Tracing):
        return None
    sequence = sequences[0]
    if sequence.lease is not None and all(
        source is target
        for source, target in zip(sequence.cache, sequence.lease_slots, strict=True)
    ):
        return sequence.lease
    if not sequence.cache or not all(layer.is_fixable for layer in sequence.cache):
        return None
    sequence.lease = compiled_decode(
        plan_of(model, screened=screened), sequence.cache, sequence.capacity
    )
    sequence.lease_slots = tuple(sequence.cache)
    return sequence.lease


def _shared_step(
    model: LanguageModel[LayerCache], sequences: Sequence[BatchSequence]
) -> list[list[int]]:
    """Every slot that does not draft, through one forward."""
    count = len(sequences)
    drawn_ids = tuple(sequence.pending for sequence in sequences)
    ids = drawn_ids[0][None, None] if count == 1 else mx.stack(drawn_ids)[:, None]
    caches = [sequence.cache for sequence in sequences]
    unfiltered_greedy = all(
        sequence.sampler is greedy and sequence.penalty is None and sequence.constraint is None
        for sequence in sequences
    )
    lease = _lease(model, sequences, screened=unfiltered_greedy) if count == 1 else None
    if lease is None and count > 1:
        for sequence in sequences:
            sequence.lease = None
            sequence.lease_slots = ()
    generic: SlotBucket | None = None
    if lease is None:
        room = max(sequence.capacity for sequence in sequences)
        needed = max(sequence.cache[0].offset for sequence in sequences) + 1
        if needed >= room:
            room = fit(needed)
        generic = _generic_bucket(model, caches, room, screened=unfiltered_greedy)
        if generic is not None:
            for sequence in sequences:
                sequence.capacity = room
    following: list[mx.array] = []
    if lease is not None:
        # The row the lease hands back is logits, or — when nothing but greedy will read it
        # — the screened stand-in whose argmax is theirs. Either way it goes through the
        # sampler below, which for this branch is `greedy` and nothing else.
        logits = lease(ids[0])
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
    _close(sequence)
    if sequence.owed and not sequence.finished:
        sequence.pending = mx.array(sequence.owed.pop(0))
    return emitted


def _close(sequence: BatchSequence) -> None:
    """The span a step just closed, or the last of them when the sequence retires.

    Between steps and nowhere else: the forward has fed `pending` and the commit has recorded
    it, so the rows are exactly `tokens` — which is also why the count is host-side and this
    costs no synchronization per token.
    """
    prefix = sequence.prefix
    if prefix is None:
        return
    rows = len(sequence.tokens)
    if sequence.finished or rows // prefix.chain.span > prefix.covered // prefix.chain.span:
        prefix.commit(sequence.tokens, sequence.cache, rows)


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
