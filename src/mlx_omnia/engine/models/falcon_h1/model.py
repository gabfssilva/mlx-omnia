import itertools
from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import Attending
from mlx_omnia.engine.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.engine.models.falcon_h1.config import FalconH1Config
from mlx_omnia.engine.models.falcon_h1.layers.block import FalconH1Layer, FalconH1Trunk
from mlx_omnia.engine.models.falcon_h1.layers.mamba import Recurring


class FalconH1Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class FalconH1(nn.Module):
    continuous_batching = True

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.model = FalconH1Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        """Two entries per block, interleaved: the mamba half then the attention half.
        A composite would have to be adapted as one thing; two halves are each already
        a shape the ragged batcher knows."""
        caches: list[LayerCache] = []
        for _ in range(self.config.num_hidden_layers):
            caches.append(DeltaCache())
            caches.append(KVCache())
        return caches

    def activations(
        self, ids: mx.array, cache: Sequence[FalconH1Layer] | None = None
    ) -> FalconH1Activations:
        layers: Sequence[FalconH1Layer] = self.make_cache() if cache is None else cache
        x = self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, (mamba_cache, kv_cache) in zip(
            self.model.layers, itertools.batched(layers, 2, strict=True), strict=True
        ):
            assert isinstance(mamba_cache, Recurring)
            assert isinstance(kv_cache, KVCache | Attending)
            x = block(x, mamba_cache, kv_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return FalconH1Activations(embedded, blocks, normed, logits)

    def __call__(self, ids: mx.array, cache: Sequence[FalconH1Layer] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
