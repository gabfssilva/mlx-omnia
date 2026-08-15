from collections.abc import Callable, Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.batching import BatchedKVCache, batch
from mlx_omnia.engine.checkpoint import wire_resident
from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, LayerCache, RingKVCache, fit
from mlx_omnia.engine.core.decode import Buckets, DecodePlan, SlotBucket, compiled_decode
from mlx_omnia.engine.core.kernels.attention.sdpa import SCALED_DOT_PRODUCT_ATTENTION
from mlx_omnia.engine.core.kernels.lm_head.argmax import (
    Int5Planes,
    int5_planes,
    lm_head_argmax_applies,
    lm_head_argmax_row,
)
from mlx_omnia.engine.core.patch import uses
from mlx_omnia.engine.models.laguna.config import FULL, SLIDING, LagunaConfig
from mlx_omnia.engine.models.laguna.layers.attention import ATLASES
from mlx_omnia.engine.models.laguna.layers.block import LagunaTrunk
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe


class LagunaActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


type LagunaCache = KVCache | FixedKVCache | RingKVCache

_BUCKETS = (2, 4, 8)


def _bucket_size(count: int) -> int:
    """The smallest compiled batch a live count fits into."""
    for size in _BUCKETS:
        if count <= size:
            return size
    raise ValueError(f"batch decode takes at most {_BUCKETS[-1]} sequences, got {count}")


def _pad_rows(ids: mx.array, size: int) -> mx.array:
    missing = size - ids.shape[0]
    if missing <= 0:
        return ids
    return mx.concatenate([ids, mx.repeat(ids[-1:], missing, axis=0)])


def _stores(slots: Sequence[LayerCache]) -> list[FixedKVCache | RingKVCache]:
    """The promoted slots, narrowed back to the two shapes a Laguna decode graph reads."""
    narrowed: list[FixedKVCache | RingKVCache] = []
    for layer in slots:
        assert isinstance(layer, FixedKVCache | RingKVCache)
        narrowed.append(layer)
    return narrowed


def _batched(slots: Sequence[Sequence[FixedKVCache | RingKVCache]]) -> list[BatchedKVCache]:
    """Every Laguna layer is a KV layer, so every adapter `batch` builds is a KV one."""
    adapters: list[BatchedKVCache] = []
    for layer in batch(slots):
        assert isinstance(layer, BatchedKVCache)
        adapters.append(layer)
    return adapters


@uses(SCALED_DOT_PRODUCT_ATTENTION)
class Laguna(nn.Module):
    continuous_batching = True

    _single_buckets: Buckets
    _batch_buckets: Buckets
    _single_live: dict[str, tuple[LayerCache, ...]]
    _batch_dead: dict[tuple[str, int, int, int], list[LagunaCache]]
    _head_planes_cache: Int5Planes | None

    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        object.__setattr__(self, "_single_buckets", Buckets())
        object.__setattr__(self, "_batch_buckets", Buckets())
        object.__setattr__(self, "_single_live", {})
        object.__setattr__(self, "_batch_dead", {})
        object.__setattr__(self, "_head_planes_cache", None)
        self.config = config
        self.model = LagunaTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | RingKVCache]:
        """Sliding layers get a fixed ring: constant shape per step is what lets the fused
        sliding reader own the append, and what keeps the decode graph from being rebuilt
        around a growing slice.

        Held back with the fused sliding reader: correct only once the reader owns the
        append, and the per-step slice writes here cost more than the growing cache."""
        return [KVCache() for _ in self.config.layer_types]

    def compile_decode(
        self, cache: list[KVCache | FixedKVCache | RingKVCache], capacity: int | None = None
    ) -> Callable[[mx.array], mx.array]:
        """Promote a completed prefill cache and compile one-token forwards."""
        return self._compile_decode(cache, self._room(cache, capacity), argmax_only=False)

    def compile_greedy_decode(
        self, cache: list[KVCache | FixedKVCache | RingKVCache], capacity: int | None = None
    ) -> Callable[[mx.array], mx.array]:
        return self._compile_decode(cache, self._room(cache, capacity), argmax_only=True)

    @staticmethod
    def _room(cache: list[KVCache | FixedKVCache | RingKVCache], capacity: int | None) -> int:
        """4096 unless the prompt outgrew it — a multi-turn conversation walks in past it —
        and then the smallest fitting bucket, since the fixed buffer is read whole by every
        step's attention."""
        return capacity if capacity is not None else max(4096, fit(cache[0].rows))

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
    ) -> SlotBucket:
        """A resident single-sequence graph, dropped the moment the live cache is another
        request's. Reloading the slots would keep the graph but not the wrapper's `base`,
        and a decode that resumes another request's step counter writes the wrong rows."""
        mode = "greedy" if greedy else "logits"
        live = self._single_live.get(mode)
        if live is not None and not all(
            source is target for source, target in zip(cache, live, strict=True)
        ):
            self._single_buckets.clear()
            self._single_live.clear()

        def build(_pending: Sequence[list[LayerCache]], room: int) -> SlotBucket:
            decode = self._compile_decode(cache, room, argmax_only=greedy)
            return SlotBucket((tuple(cache),), decode)

        view: list[list[LayerCache]] = [[*cache]]
        bucket = self._single_buckets.lease(mode, view, capacity, build)
        self._single_live[mode] = tuple(cache)
        return bucket

    def compile_batch_decode(
        self,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        capacity: int = 4096,
    ) -> Callable[[mx.array], mx.array]:
        return self._build_batch(caches, capacity, project_head=True).decode

    def _lease_batch(
        self,
        caches: Sequence[list[LagunaCache]],
        capacity: int,
        *,
        project_head: bool,
    ) -> SlotBucket:
        mode = "logits" if project_head else "hidden"
        count = len(caches)
        size = _bucket_size(count)
        padded: list[list[LagunaCache]] = [
            *caches,
            *(self._dead_slot(mode, size, capacity, row) for row in range(count, size)),
        ]
        views: list[list[LayerCache]] = [[*sequence] for sequence in padded]

        def build(_pending: Sequence[list[LayerCache]], room: int) -> SlotBucket:
            return self._build_batch(padded, room, project_head=project_head)

        bucket = self._batch_buckets.lease(mode, views, capacity, build)
        for sequence, slot in zip(padded, bucket.slots, strict=True):
            sequence[:] = _stores(slot)
        return bucket

    def _dead_slot(
        self, mode: str, size: int, capacity: int, row: int
    ) -> list[LagunaCache]:
        """A bucket row no sequence occupies. It is never loaded from a live sequence: one
        token of its own is enough to give the row a position, and every layer's validity
        mask reads that position, so the row attends its own stale state and the decode
        drops the result."""
        key = (mode, size, capacity, row)
        dead: list[LagunaCache] | None = self._batch_dead.get(key)
        if dead is None:
            dead = [*self.make_cache()]
            self._activations(mx.array([[0]]), dead, project_head=False)
            mx.eval(*(tensor for layer in dead for tensor in layer.tensors))
            self._batch_dead[key] = dead
        return dead

    def _build_batch(
        self,
        caches: Sequence[list[LagunaCache]],
        capacity: int,
        *,
        project_head: bool,
    ) -> SlotBucket:
        if len(caches) not in _BUCKETS:
            raise ValueError(f"compiled batch size must be one of {_BUCKETS}, got {len(caches)}")
        if any(len(sequence) != len(self.config.layer_types) for sequence in caches):
            raise ValueError("batch decode cache count does not match the model")
        offsets = [sequence[0].rows for sequence in caches]
        if max(offsets) >= capacity:
            raise ValueError(
                f"prompt length {max(offsets)} does not fit compiled capacity {capacity}"
            )
        slots = self._make_batch_slots(caches, capacity)
        stores = _batched(slots)
        state = [layer.state for sequence in slots for layer in sequence]

        for block in self.model.layers:
            block.self_attn.angles(max(offsets))
            block.self_attn.kernels()
            block.kernels()
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

        return SlotBucket(
            tuple(tuple(slot) for slot in slots),
            mx.compile(forward, inputs=[ATLASES, state], outputs=state),
        )

    def _make_batch_slots(
        self,
        caches: Sequence[Sequence[LayerCache]],
        capacity: int,
    ) -> list[list[FixedKVCache | RingKVCache]]:
        slots = [self._promote_slot(sequence, capacity) for sequence in caches]
        mx.eval(*(tensor for slot in slots for layer in slot for tensor in layer.tensors))
        return slots

    def _promote_slot(
        self,
        sequence: Sequence[LayerCache],
        capacity: int,
    ) -> list[FixedKVCache | RingKVCache]:
        """SLIDING layers get a ring, FULL layers a fixed buffer at `capacity`. The family's
        policy, not the core's: which layer slides is a property of the architecture."""
        slot: list[FixedKVCache | RingKVCache] = []
        for source, kind in zip(sequence, self.config.layer_types, strict=True):
            assert isinstance(source, KVCache | FixedKVCache | RingKVCache)
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
        return slot

    def batch_decode(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> mx.array:
        bucket = self._lease_batch(caches, capacity, project_head=True)
        return bucket.decode(_pad_rows(ids, len(bucket.slots)))[: len(caches)]

    def batch_greedy(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> tuple[mx.array, ...]:
        bucket = self._lease_batch(caches, capacity, project_head=False)
        count = len(caches)
        hidden = bucket.decode(_pad_rows(ids, len(bucket.slots)))[:count, -1, :]
        planes = self._head_planes()
        if planes is None:
            logits = (
                self.model.embed_tokens.as_linear(hidden)
                if self.config.tie_word_embeddings
                else self.lm_head(hidden)
            )
            tokens = mx.argmax(logits, axis=-1)
            return tuple(tokens[index] for index in range(count))
        return tuple(
            mx.argmax(lm_head_argmax_row(hidden[index], self.lm_head.weight, planes))
            for index in range(count)
        )

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
        head_planes = self._head_planes() if argmax_only else None

        def promote(current: list[LayerCache], fitting: int) -> list[LayerCache]:
            slot = self._make_batch_slots([current], fitting)[0]
            cache[:] = slot
            return [*slot]

        def prepare(slots: Sequence[LayerCache]) -> None:
            for layer, layer_cache in zip(self.model.layers, _stores(slots), strict=True):
                layer.self_attn.angles(slots[0].rows)
                layer.self_attn.prepare_decode(layer_cache)
                layer.kernels()
                if isinstance(layer.mlp, LagunaSparseMoe):
                    layer.mlp.step_applies()
            wire_resident()

        def step(ids: mx.array, slots: Sequence[LayerCache], mask: mx.array | None) -> mx.array:
            # The core's validity mask is unused: Laguna builds its own inside `_activations`,
            # where the ring's mask is the ring's own graph-visible position.
            del mask
            return self._activations(
                ids[None], _stores(slots), project_head=head_planes is None
            ).logits[:, -1, :]

        plan = DecodePlan(
            step=step,
            promote=promote,
            prepare=prepare,
            captures=lambda: [ATLASES],
        )
        layers: list[LayerCache] = [*cache]
        compiled = compiled_decode(plan, layers, capacity)
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
        cached = self._head_planes_cache
        if cached is not None:
            return cached
        weight = self.lm_head.weight
        vocab, hidden = weight.shape
        if not lm_head_argmax_applies(vocab, hidden, rows=1, dtype=weight.dtype):
            return None
        planes = int5_planes(weight)
        if planes is not None:
            mx.eval(*planes)
            object.__setattr__(self, "_head_planes_cache", planes)
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
        if not isinstance(offset, int):
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
