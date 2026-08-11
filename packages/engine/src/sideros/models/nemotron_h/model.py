from typing import NamedTuple, assert_never

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache, KVCache, LayerCache
from sideros.models.nemotron_h.config import NemotronHConfig
from sideros.models.nemotron_h.layers.block import NemotronHBlock


class NemotronHActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class NemotronHTrunk(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NemotronHBlock(config, kind) for kind in config.pattern]
        self.norm_f = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


class NemotronH(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = NemotronHTrunk(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        caches: list[LayerCache] = []
        for kind in self.config.pattern:
            match kind:
                case "M":
                    caches.append(DeltaCache())
                case "*":
                    caches.append(KVCache())
                case "E" | "-":
                    caches.append(LayerCache())
                case _:
                    assert_never(kind)
        return caches

    def activations(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> NemotronHActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.backbone.embeddings(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.backbone.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.backbone.norm_f(x)
        return NemotronHActivations(embeddings, blocks, normed, self.lm_head(normed))

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
