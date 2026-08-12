import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.afmoe.config import AfmoeConfig
from mlx_omnia.engine.models.afmoe.layers.block import AfmoeBlock


class AfmoeActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class AfmoeTrunk(nn.Module):
    def __init__(self, config: AfmoeConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            AfmoeBlock(config, layer, layer_type)
            for layer, layer_type in enumerate(config.layer_types)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Afmoe(nn.Module):
    def __init__(self, config: AfmoeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AfmoeTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def embed(self, ids: mx.array) -> mx.array:
        embedded = self.model.embed_tokens(ids)
        if not self.config.mup_enabled:
            return embedded
        return embedded * math.sqrt(self.config.hidden_size)

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> AfmoeActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.embed(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return AfmoeActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
