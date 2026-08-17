from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import LanguageModel
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.models.qwen3.config import Qwen3Config, Qwen3MoEConfig
from mlx_omnia.engine.models.qwen3.layers.block import Qwen3Block, Qwen3MoEBlock


class Qwen3Trunk(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Qwen3(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> Qwen3Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return Qwen3Activations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


class Qwen3MoETrunk(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3MoEBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3MoEActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Qwen3MoE(nn.Module, LanguageModel[LayerCache]):

    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoETrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> Qwen3MoEActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return Qwen3MoEActivations(blocks, self.head(normed))

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
