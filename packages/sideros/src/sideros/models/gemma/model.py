import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.gemma.config import GemmaConfig
from sideros.models.gemma.layers.block import GemmaBlock


class GemmaActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class GemmaTrunk(nn.Module):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GemmaBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Gemma(nn.Module):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GemmaTrunk(config)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def embed(self, ids: mx.array) -> mx.array:
        embedded = self.model.embed_tokens(ids)
        scale = mx.array(math.sqrt(self.config.hidden_size), mx.float32).astype(embedded.dtype)
        return embedded * scale

    def head(self, normed: mx.array) -> mx.array:
        return self.model.embed_tokens.as_linear(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> GemmaActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.embed(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return GemmaActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
