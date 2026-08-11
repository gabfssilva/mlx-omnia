from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.models.bailing_hybrid.layers.block import BailingHybridTrunk


class BailingHybridActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class BailingHybrid(nn.Module):
    def __init__(self, config: BailingHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BailingHybridTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        return [KVCache() if attends else DeltaCache() for attends in self.config.attends]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> BailingHybridActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.word_embeddings(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return BailingHybridActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
