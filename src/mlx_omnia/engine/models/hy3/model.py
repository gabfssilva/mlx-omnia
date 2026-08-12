from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.hy3.config import Hy3Config
from mlx_omnia.engine.models.hy3.layers.block import Hy3Trunk


class Hy3Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Hy3(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Hy3Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.config.num_hidden_layers)]

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Hy3Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        mask: mx.array | str | None = None if length == 1 else "causal"
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, mask, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Hy3Activations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
