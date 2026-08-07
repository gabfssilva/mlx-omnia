from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from sideros.models.glm4_moe.dsa.layers.block import GlmMoEDSATrunk
from sideros.models.glm4_moe.dsa.layers.cache import DSACache


class GlmMoEDSAActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class GlmMoEDSA(nn.Module):
    def __init__(self, config: GlmMoEDSAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GlmMoEDSATrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[DSACache]:
        return [DSACache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[DSACache] | None = None
    ) -> GlmMoEDSAActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return GlmMoEDSAActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[DSACache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
