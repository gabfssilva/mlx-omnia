from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import Tracing
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from mlx_omnia.engine.models.glm4_moe.dsa.layers.block import GlmMoEDSATrunk
from mlx_omnia.engine.models.glm4_moe.dsa.layers.cache import (
    BatchedDSACache,
    DSACache,
    FixedDSACache,
)


class GlmMoEDSAActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class GlmMoEDSA(nn.Module, Tracing[LayerCache]):

    def __init__(self, config: GlmMoEDSAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GlmMoEDSATrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[DSACache]:
        return [DSACache() for _ in self.model.layers]

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. Nothing to settle and nothing to capture: this trunk resolves
        no kernel against its checkpoint, so the trace has only the caches to be handed.

        Both halves of the claim hold. Every position the forward rotates by is
        `FixedDSACache.position` — a graph tensor, read before either buffer is updated —
        and the columns a promoted buffer has not written are cut twice over: the indexer
        scores them at `-inf` before it selects, and the attention ANDs whatever the
        selection returned against the same band.
        """
        del cache
        return ()

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> GlmMoEDSAActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            if not isinstance(layer_cache, DSACache | FixedDSACache | BatchedDSACache):
                raise TypeError(f"a glm4_moe dsa forward mixes in {type(layer_cache).__name__}")
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return GlmMoEDSAActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
