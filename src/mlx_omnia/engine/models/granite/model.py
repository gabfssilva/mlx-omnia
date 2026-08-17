from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import LanguageModel
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.models.granite.config import GraniteConfig
from mlx_omnia.engine.models.granite.layers.block import GraniteBlock


class GraniteTrunk(nn.Module):
    def __init__(self, config: GraniteConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GraniteBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GraniteActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Granite(nn.Module, LanguageModel[LayerCache]):

    def __init__(self, config: GraniteConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GraniteTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return logits / self.config.logits_scaling

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> GraniteActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids) * self.config.embedding_multiplier
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return GraniteActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
