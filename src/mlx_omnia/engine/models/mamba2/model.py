from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.layers.block import Mamba2Trunk


class Mamba2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Mamba2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[DeltaCache]:
        return [DeltaCache() for _ in range(self.config.num_hidden_layers)]

    def activations(
        self,
        ids: mx.array,
        cache: list[DeltaCache] | None = None,
    ) -> Mamba2Activations:
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
        return Mamba2Activations(embedded, blocks, normed, logits)

    def __call__(
        self,
        ids: mx.array,
        cache: list[DeltaCache] | None = None,
    ) -> mx.array:
        return self.activations(ids, cache).logits
