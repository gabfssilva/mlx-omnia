from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.models.bailing_moe.config import BailingMoEConfig
from mlx_omnia.models.bailing_moe.layers.block import BailingMoEBlock


class BailingMoEActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class BailingMoETrunk(nn.Module):
    def __init__(self, config: BailingMoEConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            BailingMoEBlock(config, layer >= config.first_k_dense_replace)
            for layer in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class BailingMoE(nn.Module):
    def __init__(self, config: BailingMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BailingMoETrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[KVCache] | None = None
    ) -> BailingMoEActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.word_embeddings(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return BailingMoEActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
