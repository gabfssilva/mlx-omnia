from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.engine.core.attention import DenseActivations, DenseModel
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.models.smollm3.config import SmolLM3Config


class SmolLM3(DenseModel):
    """The house's dense decoder, leaf for leaf — the delta is one rotation bit per layer."""


    def __init__(self, config: SmolLM3Config) -> None:
        super().__init__(config.dense, config.rotary)

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> DenseActivations:
        """The core dense forward, widened to the ragged batch caches: the blocks only
        pass the cache through, so the arithmetic is the trunk's own."""
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return DenseActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
