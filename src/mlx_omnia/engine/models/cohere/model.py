from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.cohere.config import CohereConfig
from mlx_omnia.engine.models.cohere.layers.block import CohereBlock


class CohereTrunk(nn.Module):
    def __init__(self, config: CohereConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [CohereBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias
        )


class CohereActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Cohere(nn.Module):
    def __init__(self, config: CohereConfig) -> None:
        super().__init__()
        self.config = config
        self.model = CohereTrunk(config)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        return self.model.embed_tokens.as_linear(normed) * self.config.logit_scale

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> CohereActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return CohereActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
