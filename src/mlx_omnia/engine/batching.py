"""Continuous batching primitives."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx

from mlx_omnia.engine.core.attend import AttentionMask, KVStore
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, RingKVCache
from mlx_omnia.engine.core.prefill import prefill
from mlx_omnia.engine.generate import Constraint, Meter, Penalty, Sampler, greedy

__all__ = ["BatchSequence", "BatchedKVCache", "batch", "prepare_batch_sequence", "step"]


@runtime_checkable
class BatchModel(Protocol):
    continuous_batching: bool

    def make_cache(self) -> list[KVCache]: ...

    def __call__(self, ids: mx.array, cache: Sequence[KVStore]) -> mx.array: ...


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
                )
            )
        return mx.concatenate(attended)


def batch(caches: Sequence[Sequence[KVCache]]) -> list[KVStore]:
    """Transpose per-sequence caches into per-layer ragged batch adapters.

    Parameters
    ----------
    caches : Sequence[Sequence[KVCache]]
        Cache layers grouped by sequence.

    Returns
    -------
    list[KVStore]
        Cache adapters grouped by model layer.
    """
    return [BatchedKVCache(layers) for layers in zip(*caches, strict=True)]


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

    Returns
    -------
    BatchSequence
        State ready for the scheduler's first step.
    """
    cache = model.make_cache() if cache is None else cache
    history = mx.array(prompt)
    remaining = history[None, reused:]
    last = prefill(lambda part: model(remaining[:, part], cache), remaining.shape[1], cache)
    logits = model(remaining[:, last], cache)[:, -1, :]
    if penalty is not None:
        logits = penalty(logits, history)
    if constraint is not None:
        logits = constraint.mask(logits, max_tokens)
    pending = sampler(logits)[0]
    mx.async_eval(pending)
    if meter is not None:
        meter.prefill(len(prompt), reused)
    required = cache[0].offset + max_tokens
    capacity = (required + 255) // 256 * 256
    return BatchSequence(
        cache,
        pending,
        sampler,
        stop,
        max_tokens,
        history,
        list(prompt),
        capacity,
        penalty,
        constraint,
        meter,
        finished=max_tokens <= 0,
    )


def step(model: BatchModel, sequences: Sequence[BatchSequence]) -> list[int | None]:
    """Advance every active sequence through one shared model forward.

    Parameters
    ----------
    model : BatchModel
        Model shared by the active sequences.
    sequences : Sequence[BatchSequence]
        Active scheduler slots.

    Returns
    -------
    list[int | None]
        Emitted token per slot; ``None`` when its stop token was consumed.
    """
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
            return [_commit(sequence, sequence.pending, queued)]
    drawn_ids = tuple(sequence.pending for sequence in sequences)
    ids = drawn_ids[0][None, None] if count == 1 else mx.stack(drawn_ids)[:, None]
    caches = [sequence.cache for sequence in sequences]
    bucketed = count in (2, 4)
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
    else:
        logits = model(ids, batch(caches))[:, -1, :]
    if logits is not None:
        following = []
        for index, sequence in enumerate(sequences):
            row = logits[index : index + 1]
            if sequence.penalty is not None:
                sequence.history = mx.concatenate([sequence.history, sequence.pending[None]])
                row = sequence.penalty(row, sequence.history)
            if sequence.constraint is not None:
                row = sequence.constraint.mask(row, sequence.remaining - 1)
            following.append(sequence.sampler(row)[0])
    mx.async_eval(*following)
    return [
        _commit(sequence, drawn, queued)
        for sequence, drawn, queued in zip(sequences, drawn_ids, following, strict=True)
    ]


def _commit(sequence: BatchSequence, drawn: mx.array, queued: mx.array) -> int | None:
    token = drawn.item()
    if not isinstance(token, int):
        raise TypeError(f"sampler returned {type(token).__name__}, expected int")
    sequence.pending = queued
    sequence.tokens.append(token)
    if token in sequence.stop:
        sequence.finished = True
        return None
    sequence.remaining -= 1
    if sequence.meter is not None:
        sequence.meter.token()
    accepted = sequence.constraint is None or sequence.constraint.accept(token)
    sequence.finished = sequence.remaining == 0 or not accepted
    return token
