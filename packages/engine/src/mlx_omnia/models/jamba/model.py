from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.models.jamba.config import JambaConfig
from mlx_omnia.models.jamba.layers.block import JambaBlock


class JambaActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class JambaTrunk(nn.Module):
    def __init__(self, config: JambaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            JambaBlock(config, attends, routes)
            for attends, routes in zip(config.attends, config.routes, strict=True)
        ]
        self.final_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Jamba(nn.Module):
    def __init__(self, config: JambaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = JambaTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        return [KVCache() if attends else DeltaCache() for attends in self.config.attends]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> JambaActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.final_layernorm(x)
        return JambaActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
