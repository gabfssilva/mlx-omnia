from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.checkpoint import wire_resident
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


@uses(SCALED_DOT_PRODUCT_ATTENTION)
class Laguna(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
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

    def _compile_decode(
        self,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        capacity: int,
        *,
        argmax_only: bool,
    ) -> Callable[[mx.array], mx.array]:
        if not cache or not all(isinstance(layer, KVCache) for layer in cache):
            raise ValueError("decode compilation requires a completed growing KV cache")
        offset = cache[0].offset
        if offset >= capacity:
            raise ValueError(f"prompt length {offset} does not fit compiled capacity {capacity}")
        promoted = [
            RingKVCache.promote(layer, self.config.sliding_window)
            if kind == SLIDING
            else FixedKVCache.promote(layer, capacity)
            for layer, kind in zip(cache, self.config.layer_types, strict=True)
        ]
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
        cache: list[KVCache | FixedKVCache | RingKVCache] | None = None,
    ) -> LagunaActivations:
        return self._activations(ids, cache, project_head=True)

    def _activations(
        self,
        ids: mx.array,
        cache: list[KVCache | FixedKVCache | RingKVCache] | None,
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
        self, ids: mx.array, cache: list[KVCache | FixedKVCache | RingKVCache] | None = None
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

    def _sliding_mask(self, length: int, offset: int) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + window`, built only where
        it is not already something cheaper. No key is old enough for the window to
        cut while `offset + length <= window`, so the band *is* the causal mask there
        — and at T=1 the single row is causal by construction, leaving `columns >
        offset - window`."""
        window = self.config.sliding_window
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)
