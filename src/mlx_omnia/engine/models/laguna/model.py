from collections.abc import Callable, Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.checkpoint import wire_resident
from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, RingKVCache
from mlx_omnia.engine.core.kernels.attention.sdpa import SCALED_DOT_PRODUCT_ATTENTION
from mlx_omnia.engine.core.kernels.lm_head.argmax import (
    Int5Planes,
    int5_planes,
    lm_head_argmax_applies,
    lm_head_argmax_row,
)
from mlx_omnia.engine.core.patch import uses
from mlx_omnia.engine.models.laguna.config import FULL, SLIDING, LagunaConfig
from mlx_omnia.engine.models.laguna.layers.attention import _ATLASES
from mlx_omnia.engine.models.laguna.layers.block import LagunaTrunk
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe

_LM_HEAD_PLANES: dict[int, tuple[mx.array, mx.array, mx.array]] = {}


class LagunaActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


type LagunaCache = KVCache | FixedKVCache | RingKVCache


class _BatchBucket(NamedTuple):
    slots: tuple[tuple[LagunaCache, ...], ...]
    decode: Callable[[mx.array], mx.array]


class _SingleBucket(NamedTuple):
    slots: tuple[LagunaCache, ...]
    decode: Callable[[mx.array], mx.array]


@uses(SCALED_DOT_PRODUCT_ATTENTION)
class Laguna(nn.Module):
    continuous_batching = True

    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        object.__setattr__(self, "_single_decodes", {})
        object.__setattr__(self, "_single_greedy_decodes", {})
        object.__setattr__(self, "_batch_decodes", {})
        object.__setattr__(self, "_batch_greedy_decodes", {})
        self.config = config
        self.model = LagunaTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | RingKVCache]:
        """Sliding layers get a fixed ring: constant shape per step is what lets the fused
        sliding reader own the append, and what keeps the decode graph from being rebuilt
        around a growing slice."""
        window = self.config.sliding_window
        # Held back with the fused sliding reader: correct only once the reader owns the
        # append, and the per-step slice writes here cost more than the growing cache.
        del window
        return [KVCache() for _ in self.config.layer_types]

    def compile_decode(
        self, cache: list[KVCache | FixedKVCache | RingKVCache], capacity: int = 4096
    ) -> Callable[[mx.array], mx.array]:
        """Promote a completed prefill cache and compile one-token forwards."""
        return self._compile_decode(cache, capacity, argmax_only=False)

    def compile_greedy_decode(
        self, cache: list[KVCache | FixedKVCache | RingKVCache], capacity: int = 4096
    ) -> Callable[[mx.array], mx.array]:
        return self._compile_decode(cache, capacity, argmax_only=True)

    def single_decode(
        self,
        ids: mx.array,
        cache: list[LagunaCache],
        *,
        capacity: int,
    ) -> mx.array:
        bucket = self._single_bucket(cache, capacity, greedy=False)
        return bucket.decode(ids)

    def single_greedy(
        self,
        ids: mx.array,
        cache: list[LagunaCache],
        *,
        capacity: int,
    ) -> mx.array:
        bucket = self._single_bucket(cache, capacity, greedy=True)
        return mx.argmax(bucket.decode(ids), axis=-1)

    def prepare_single_greedy(
        self,
        cache: list[LagunaCache],
        *,
        capacity: int,
    ) -> Callable[[mx.array], mx.array]:
        decode = self._compile_decode(cache, capacity, argmax_only=True)

        def greedy_decode(ids: mx.array) -> mx.array:
            return mx.argmax(decode(ids), axis=-1)[0]

        return greedy_decode

    def _single_bucket(
        self,
        cache: list[LagunaCache],
        capacity: int,
        *,
        greedy: bool,
    ) -> _SingleBucket:
        decodes: dict[tuple[int, int], _SingleBucket] = (
            self._single_greedy_decodes if greedy else self._single_decodes
        )
        key = (id(cache), capacity)
        bucket = decodes.get(key)
        if bucket is not None and all(
            source is target for source, target in zip(cache, bucket.slots, strict=True)
        ):
            return bucket
        decodes.clear()
        decode = self._compile_decode(cache, capacity, argmax_only=greedy)
        bucket = _SingleBucket(tuple(cache), decode)
        decodes[key] = bucket
        return bucket

    def compile_batch_decode(
        self,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        capacity: int = 4096,
    ) -> Callable[[mx.array], mx.array]:
        return self._compile_batch_forward(caches, capacity, project_head=True)

    def _compile_batch_forward(
        self,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        capacity: int,
        *,
        project_head: bool,
    ) -> Callable[[mx.array], mx.array]:
        if len(caches) not in (2, 4):
            raise ValueError(f"compiled batch size must be 2 or 4, got {len(caches)}")
        if any(len(sequence) != len(self.config.layer_types) for sequence in caches):
            raise ValueError("batch decode cache count does not match the model")
        offsets = [sequence[0].rows for sequence in caches]
        if max(offsets) >= capacity:
            raise ValueError(
                f"prompt length {max(offsets)} does not fit compiled capacity {capacity}"
            )
        slots = self._make_batch_slots(caches, capacity)
        stores = batch(slots)
        state = [
            layer.state
            for sequence in slots
            for layer in sequence
            if isinstance(layer, (FixedKVCache, RingKVCache))
        ]

        for block in self.model.layers:
            block.self_attn._angles(max(offsets))
            block.self_attn._kernels()
            block._kernels()
            if isinstance(block.mlp, LagunaSparseMoe):
                block.mlp.step_applies()
        wire_resident()
        for sequence, slot in zip(caches, slots, strict=True):
            sequence[:] = slot

        def forward(ids: mx.array) -> mx.array:
            x = self.model.embed_tokens(ids)
            for block, layer_cache in zip(self.model.layers, stores, strict=True):
                x = block(x, None, layer_cache)
            normed = self.model.norm(x)
            if not project_head:
                return normed
            if self.config.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(normed)
            return self.lm_head(normed)

        return mx.compile(forward, inputs=[_ATLASES, state], outputs=state)

    def _make_batch_slots(
        self,
        caches: Sequence[list[LagunaCache]],
        capacity: int,
    ) -> list[list[FixedKVCache | RingKVCache]]:
        slots: list[list[FixedKVCache | RingKVCache]] = []
        for sequence in caches:
            slot: list[FixedKVCache | RingKVCache] = []
            for source, kind in zip(sequence, self.config.layer_types, strict=True):
                if kind == SLIDING:
                    if isinstance(source, KVCache):
                        target = RingKVCache.promote(source, self.config.sliding_window)
                    elif isinstance(source, RingKVCache):
                        keys, values = source.fetch()
                        target = RingKVCache(self.config.sliding_window)
                        target.restore(
                            source.rows,
                            {
                                "keys": keys + mx.zeros_like(keys),
                                "values": values + mx.zeros_like(values),
                            },
                        )
                        target_keys, target_values = target.fetch()
                        target.state = [
                            target_keys,
                            target_values,
                            mx.array([source.rows], dtype=mx.int32),
                        ]
                    else:
                        raise TypeError("sliding layer requires a growing or ring KV cache")
                else:
                    if isinstance(source, RingKVCache):
                        raise TypeError("full layer cannot be restored from a ring KV cache")
                    keys, values = source.fetch()
                    rows = source.rows
                    shape = list(keys.shape)
                    shape[2] = capacity
                    fixed_keys = mx.zeros(shape, dtype=keys.dtype)
                    fixed_values = mx.zeros(shape, dtype=values.dtype)
                    fixed_keys[..., :rows, :] = keys[..., :rows, :]
                    fixed_values[..., :rows, :] = values[..., :rows, :]
                    target = FixedKVCache(fixed_keys, fixed_values, rows)
                slot.append(target)
            slots.append(slot)
        mx.eval(*(tensor for slot in slots for layer in slot for tensor in layer.tensors))
        return slots

    def batch_decode(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> mx.array:
        key = (len(caches), capacity)
        decodes: dict[tuple[int, int], _BatchBucket] = self._batch_decodes
        bucket = decodes.get(key)
        if bucket is None:
            decode = self.compile_batch_decode(caches, capacity)
            bucket = _BatchBucket(tuple(tuple(sequence) for sequence in caches), decode)
            decodes[key] = bucket
        else:
            self._load_batch_slots(caches, bucket.slots)
        return bucket.decode(ids)

    def batch_greedy(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> tuple[mx.array, ...]:
        key = (len(caches), capacity)
        decodes: dict[tuple[int, int], _BatchBucket] = self._batch_greedy_decodes
        bucket = decodes.get(key)
        if bucket is None:
            decode = self._compile_batch_forward(caches, capacity, project_head=False)
            bucket = _BatchBucket(tuple(tuple(sequence) for sequence in caches), decode)
            decodes[key] = bucket
        else:
            self._load_batch_slots(caches, bucket.slots)
        hidden = bucket.decode(ids)[:, -1, :]
        planes = self._head_planes()
        if planes is None:
            logits = (
                self.model.embed_tokens.as_linear(hidden)
                if self.config.tie_word_embeddings
                else self.lm_head(hidden)
            )
            tokens = mx.argmax(logits, axis=-1)
            return tuple(tokens[index] for index in range(hidden.shape[0]))
        return tuple(
            mx.argmax(lm_head_argmax_row(hidden[index], self.lm_head.weight, planes))
            for index in range(hidden.shape[0])
        )

    @staticmethod
    def _load_batch_slots(
        caches: Sequence[list[LagunaCache]],
        slots: tuple[tuple[LagunaCache, ...], ...],
    ) -> None:
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
                keys, values = source.fetch()
                copied_keys = keys + mx.zeros_like(keys)
                copied_values = values + mx.zeros_like(values)
                snapshots.append((source.rows, copied_keys, copied_values))
        mx.eval(*(array for _, keys, values in snapshots for array in (keys, values)))

        index = 0
        for sequence, slot in zip(caches, slots, strict=True):
            for target in slot:
                rows, keys, values = snapshots[index]
                index += 1
                if isinstance(target, FixedKVCache):
                    target.restore(
                        rows,
                        {"keys": keys[..., :rows, :], "values": values[..., :rows, :]},
                    )
                elif isinstance(target, RingKVCache):
                    assert target.state is not None
                    target.state[0] = keys
                    target.state[1] = values
                    target.state[2] = mx.array([rows], dtype=mx.int32)
                    target.offset = rows
                else:
                    raise TypeError(f"compiled slot cannot use {type(target).__name__}")
            sequence[:] = slot
        mx.eval(*(tensor for slot in slots for layer in slot for tensor in layer.tensors))

    def _compile_decode(
        self,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        capacity: int,
        *,
        argmax_only: bool,
    ) -> Callable[[mx.array], mx.array]:
        if len(cache) != len(self.config.layer_types):
            raise ValueError("decode cache count does not match the model")
        offset = cache[0].rows
        if offset >= capacity:
            raise ValueError(f"prompt length {offset} does not fit compiled capacity {capacity}")
        promoted = self._make_batch_slots([cache], capacity)[0]
        cache[:] = promoted
        state = [layer.state for layer in promoted]

        head_planes = self._head_planes() if argmax_only else None

        def forward(ids: mx.array) -> mx.array:
            return self._activations(ids[None], promoted, project_head=head_planes is None).logits[
                :, -1, :
            ]

        for layer, layer_cache in zip(self.model.layers, promoted, strict=True):
            layer.self_attn._angles(offset)
            layer.self_attn._prepare_decode(layer_cache)
            layer._kernels()
            if isinstance(layer.mlp, LagunaSparseMoe):
                layer.mlp.step_applies()
        wire_resident()
        # Weights and the strategies' precomputed banks are deliberately NOT captured:
        # mx.compile swaps arrays inside captured containers for tracers in place, and a
        # frozen strategy field reads its original object — which trips the uncaptured-
        # input check the moment that object is also in the capture list. Left out, they
        # bake into the trace as constants, which is what a decode closure wants; only
        # what changes across steps (the ring/fixed cache state, a regrown atlas) is
        # captured, and both are read through live containers at trace time.
        inputs = [_ATLASES, state]
        compiled = mx.compile(
            forward,
            inputs=inputs,
            outputs=state,
        )
        if head_planes is None:
            return compiled

        def greedy_forward(ids: mx.array) -> mx.array:
            return self._greedy_logits(compiled(ids), head_planes)

        return greedy_forward

    def activations(
        self,
        ids: mx.array,
        cache: Sequence[KVStore] | None = None,
    ) -> LagunaActivations:
        return self._activations(ids, cache, project_head=True)

    def _activations(
        self,
        ids: mx.array,
        cache: Sequence[KVStore] | None,
        *,
        project_head: bool,
    ) -> LagunaActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if SLIDING in self.config.layer_types:
            ring = next(
                (
                    layer
                    for layer, kind in zip(cache, self.config.layer_types, strict=True)
                    if kind == SLIDING and isinstance(layer, RingKVCache)
                ),
                None,
            )
            sliding = (
                self._ring_mask(ring) if ring is not None else self._sliding_mask(length, offset)
            )

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            x = block(x, full if kind == FULL else sliding, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if not project_head:
            logits = normed
        elif self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LagunaActivations(blocks, logits)

    def _head_planes(self) -> Int5Planes | None:
        if self.config.tie_word_embeddings:
            return None
        weight = self.lm_head.weight
        vocab, hidden = weight.shape
        if not lm_head_argmax_applies(vocab, hidden, rows=1, dtype=weight.dtype):
            return None
        key = id(weight)
        if cached := _LM_HEAD_PLANES.get(key):
            return Int5Planes(*cached)
        planes = int5_planes(weight)
        if planes is not None:
            mx.eval(*planes)
            _LM_HEAD_PLANES[key] = tuple(planes)
        return planes

    def _greedy_logits(self, x: mx.array, planes: Int5Planes | None = None) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(x)
        planes = planes if planes is not None else self._head_planes()
        if planes is None:
            return self.lm_head(x)
        return lm_head_argmax_row(x, self.lm_head.weight, planes).reshape(
            *x.shape[:-1], self.config.vocab_size
        )

    def __call__(
        self, ids: mx.array, cache: Sequence[KVStore] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

    def _ring_mask(self, ring: RingKVCache) -> mx.array | None:
        """The ring's own rows, for a reader that attends the whole buffer instead of a
        growing slice. Slot `j` holds the absolute position `a` with `a % window == j`, so
        the filled slots are `0..position` while the ring is still short of a full window —
        and the order of the rest does not matter, the keys were rotated on the way in.
        A ring the prefill already filled stays full, and needs no mask at all."""
        if ring.offset >= ring.window:
            return None
        return (mx.arange(ring.window) <= ring.position).reshape(1, 1, 1, ring.window)

    def _sliding_mask(self, length: int, offset: int | mx.array) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + window`, built only where
        it is not already something cheaper. No key is old enough for the window to
        cut while `offset + length <= window`, so the band *is* the causal mask there
        — and at T=1 the single row is causal by construction, leaving `columns >
        offset - window`."""
        window = self.config.sliding_window
        if isinstance(offset, mx.array):
            keys = int(mx.max(offset).item()) + length
            columns = mx.arange(keys)[None, None, None, :]
            rows = offset[:, None] + mx.arange(length)[None, :]
            positions = rows[:, None, :, None]
            return (positions >= columns) & (positions < columns + window)
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)
