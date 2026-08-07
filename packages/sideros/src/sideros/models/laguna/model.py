from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.laguna.config import FULL, SLIDING, LagunaConfig
from sideros.models.laguna.layers.block import LagunaTrunk


class LagunaActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Laguna(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LagunaTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def activations(
        self, ids: mx.array, cache: list[KVCache] | None = None
    ) -> LagunaActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if SLIDING in self.config.layer_types:
            sliding = self._sliding_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            x = block(x, full if kind == FULL else sliding, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LagunaActivations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits

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
