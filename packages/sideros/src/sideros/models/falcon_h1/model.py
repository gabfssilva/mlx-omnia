from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.models.falcon_h1.config import FalconH1Config
from sideros.models.falcon_h1.layers.block import FalconH1Trunk
from sideros.models.falcon_h1.layers.cache import FalconH1LayerCache


class FalconH1Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class FalconH1(nn.Module):
    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.model = FalconH1Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[FalconH1LayerCache]:
        return [FalconH1LayerCache() for _ in range(self.config.num_hidden_layers)]

    def activations(
        self, ids: mx.array, cache: list[FalconH1LayerCache] | None = None
    ) -> FalconH1Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return FalconH1Activations(embedded, blocks, normed, logits)

    def __call__(self, ids: mx.array, cache: list[FalconH1LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
