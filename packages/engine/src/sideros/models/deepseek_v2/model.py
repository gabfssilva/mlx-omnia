from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.rope import yarn
from sideros.models.deepseek_v2.config import DeepseekV2Config
from sideros.models.deepseek_v2.layers.block import DeepseekV2Block


class DeepseekV2Trunk(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        rope = yarn(config.qk_rope_head_dim, config.rope_theta, config.rope_scaling)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV2Block(config, rope, routes) for routes in config.routes]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class DeepseekV2(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        self.config = config
        self.model = DeepseekV2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[KVCache] | None = None
    ) -> DeepseekV2Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return DeepseekV2Activations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
