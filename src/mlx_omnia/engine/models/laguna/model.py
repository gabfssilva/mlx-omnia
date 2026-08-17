from collections.abc import Iterable, Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.checkpoint import wire_resident
from mlx_omnia.engine.core.api import Screened, Tracing
from mlx_omnia.engine.core.cache import (
    FixedKVCache,
    KVCache,
    LayerCache,
    RingKVCache,
)
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

def _stores(slots: Iterable[LayerCache]) -> list[FixedKVCache | RingKVCache]:
    """The promoted slots, narrowed back to the two shapes a Laguna decode graph reads."""
    narrowed: list[FixedKVCache | RingKVCache] = []
    for layer in slots:
        assert isinstance(layer, FixedKVCache | RingKVCache)
        narrowed.append(layer)
    return narrowed


@uses(SCALED_DOT_PRODUCT_ATTENTION)
class Laguna(nn.Module, Tracing[LayerCache], Screened[LayerCache]):

    _head_planes_cache: Int5Planes | None

    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        object.__setattr__(self, "_head_planes_cache", None)
        self.config = config
        self.model = LagunaTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        """A growing buffer per layer, and the sliding ones carry their window.

        Which layer slides is the architecture's, so it is declared here. What the window
        then buys is the cache's own: `layout` can say how deep a resume has to read, and
        `fixed` promotes into a ring instead of a buffer. Both used to be this class's
        answers, which made a model declare something about its state.

        The prefill still keeps the whole history — the mask is what makes a layer sliding
        — and the ring arrives on promotion, where a constant shape per step is what lets
        the fused sliding reader own the append."""
        return [
            KVCache(window=self.config.sliding_window if kind == SLIDING else None)
            for kind in self.config.layer_types
        ]

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. Everything a traced step cannot do for itself, once per build.

        The atlas is grown and evaluated, each layer's attention resolves against the shape
        it will actually read, and the fused kernels and the MoE's step applies are chosen.
        All of it is host-side work — a dict write, an `mx.eval`, a branch on a Python value
        — and a trace runs its Python exactly once, so what is not settled here is settled
        wrong.

        The atlas goes back as a capture because it is the one thing here that the graph
        *reads* and something outside it rewrites: a conversation walking past `_ATLAS_ROWS`
        grows the table, and a trace holding the old array would rotate every later token by
        an angle off the end of it.
        """
        # Sized to the capacity and not to the rows written: inside the graph the position is
        # a tensor and the rotation reads the table by `mx.take`, so every column this buffer
        # could reach has to be in it before the trace, not by the time it gets there.
        span = cache[0].span
        for layer, layer_cache in zip(self.model.layers, cache, strict=True):
            layer.self_attn.angles(span - 1)
            if isinstance(layer_cache, FixedKVCache | RingKVCache):
                # One row, read raw: the attention resolves against the buffer it will
                # actually index, which is what keeps the fused single-row kernel on.
                layer.self_attn.prepare_decode(layer_cache)
            else:
                # A ragged batch. The adapter owns the read and the per-row mask, so what is
                # left to settle is the kernels the block chooses.
                layer.self_attn.kernels()
            layer.kernels()
            if isinstance(layer.mlp, LagunaSparseMoe):
                layer.mlp.step_applies()
        wire_resident()
        return [ATLASES]

    def screened(self, ids: mx.array, cache: Sequence[LayerCache]) -> mx.array:
        """`core.api.Screened`. The greedy id off a screened head instead of a projection.

        `lm_head` is bf16 `[vocab, hidden]` — 411 MB on Laguna-XS, against blocks that are
        4-bit — and a greedy step reads one number out of it. The chain in
        `core.kernels.lm_head` reads the coarse int5 planes for the whole vocabulary and the
        full bf16 rows only for the candidates a certificate cannot rule out.

        Row by row, because the chain is a single-token head: `lm_head_argmax_applies`
        answers for `rows=1` and nothing else. Worth 1.108x on one row and 1.066x on two;
        by four it is a wash, and the loop is what stops paying.
        """
        planes = self._head_planes()
        if planes is None:
            return self(ids, cache)
        hidden = self._activations(ids, cache, project_head=False).logits[:, -1, :]
        rows = [
            lm_head_argmax_row(hidden[index], self.lm_head.weight, planes)
            for index in range(hidden.shape[0])
        ]
        return mx.stack(rows)[:, None, :]

    def activations(
        self,
        ids: mx.array,
        cache: Sequence[LayerCache] | None = None,
    ) -> LagunaActivations:
        return self._activations(ids, cache, project_head=True)

    def _activations(
        self,
        ids: mx.array,
        cache: Sequence[LayerCache] | None,
        *,
        project_head: bool,
    ) -> LagunaActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if SLIDING in self.config.layer_types:
            slider = next(
                layer
                for layer, kind in zip(cache, self.config.layer_types, strict=True)
                if kind == SLIDING
            )
            sliding = (
                self._ring_mask(slider)
                if isinstance(slider, RingKVCache)
                else self._sliding_mask(length, slider.offset, span=slider.span)
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
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
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

    def _sliding_mask(
        self, length: int, offset: int | mx.array, *, span: int
    ) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + window`, built only where
        it is not already something cheaper. No key is old enough for the window to
        cut while `offset + length <= window`, so the band *is* the causal mask there
        — and at T=1 the single row is causal by construction, leaving `columns >
        offset - window`."""
        window = self.config.sliding_window
        if not isinstance(offset, int):
            # `span` and not `max(offset)`: the positions here are the graph's, and
            # evaluating one is what a compiled decode cannot do. The layer answers with a
            # static column count because buffer shapes do not move inside a step.
            keys = span
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
