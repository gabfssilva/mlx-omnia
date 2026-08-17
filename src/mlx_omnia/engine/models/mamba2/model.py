from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import Tracing
from mlx_omnia.engine.core.cache import DeltaCache, LayerCache
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.layers.block import Mamba2Trunk
from mlx_omnia.engine.models.mamba2.layers.ssd import Recurring

type Mamba2Layer = LayerCache | Recurring
"""A layer's cache, alone or standing for one row each — read through `window`/`state`."""


class Mamba2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Mamba2(nn.Module, Tracing[LayerCache]):

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Mamba2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        return [DeltaCache() for _ in range(self.config.num_hidden_layers)]

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`, and there is nothing to settle: the forward is ops all the way
        down and resolves nothing host-side.

        What the claim buys here is the lease. The trunk is all delta — the state is
        size-free, so the capacity the core threads through promotion is ignored, and with no
        attention layer there is no buffer whose unwritten columns could need masking. Both
        halves of `Tracing` are vacuously true, which is exactly the case that used to ship a
        whole compiled decode to say so.
        """
        del cache
        return ()

    def activations(
        self,
        ids: mx.array,
        cache: Sequence[Mamba2Layer] | None = None,
    ) -> Mamba2Activations:
        layers: Sequence[Mamba2Layer] = self.make_cache() if cache is None else cache
        x = self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, layers, strict=True):
            assert isinstance(layer_cache, Recurring)
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
        cache: Sequence[Mamba2Layer] | None = None,
    ) -> mx.array:
        return self.activations(ids, cache).logits
