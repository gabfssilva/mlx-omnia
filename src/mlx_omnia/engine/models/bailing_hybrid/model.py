from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import Tracing
from mlx_omnia.engine.core.cache import DeltaCache, LayerCache
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.engine.models.bailing_hybrid.layers.block import (
    BailingHybridLayer,
    BailingHybridTrunk,
)
from mlx_omnia.engine.models.bailing_hybrid.layers.cache import LatentKVCache


class BailingHybridActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class BailingHybrid(nn.Module, Tracing[LayerCache]):

    def __init__(self, config: BailingHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BailingHybridTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        return [LatentKVCache() if attends else DeltaCache() for attends in self.config.attends]

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. Nothing to settle: this trunk resolves no kernels lazily and
        reads no resident table, so a trace of its ordinary forward is the forward.

        The two claims it does make. The latent layers read `FixedKVCache.position` rather
        than `offset` and cut their own columns through `LayerCache.readable`
        (`BailingHybridLatentAttention.__call__`); the KDA layers reach their window and
        state through `FixedDeltaCache`'s properties over the graph container, which is
        where a trace can see them written.
        """
        del cache
        return ()

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: Sequence[BailingHybridLayer] | None = None
    ) -> BailingHybridActivations:
        layers: Sequence[BailingHybridLayer] = self.make_cache() if cache is None else cache
        x = self.model.word_embeddings(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, layers, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return BailingHybridActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(
        self, ids: mx.array, cache: Sequence[BailingHybridLayer] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits
