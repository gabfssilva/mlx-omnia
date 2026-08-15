from collections.abc import Callable, Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import DeltaCache, FixedDeltaCache, LayerCache
from mlx_omnia.engine.core.decode import DecodePlan, compiled_decode
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


class Mamba2(nn.Module):
    continuous_batching = True

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Mamba2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        return [DeltaCache() for _ in range(self.config.num_hidden_layers)]

    def compile_decode(
        self,
        cache: list[LayerCache],
        capacity: int | None = None,
    ) -> Callable[[mx.array], mx.array]:
        """Promote a completed prefill cache and compile one-token forwards.

        The trunk is all delta: the state is size-free, so the capacity the core threads
        through promotion is ignored, and with no full-attention layer there is no anchor
        and no validity mask to read.
        """

        def promote(layers: list[LayerCache], fitting: int) -> list[LayerCache]:
            del fitting
            return [
                FixedDeltaCache.promote(layer)
                if isinstance(layer, DeltaCache) and not isinstance(layer, FixedDeltaCache)
                else layer
                for layer in layers
            ]

        def step(ids: mx.array, slots: Sequence[LayerCache], mask: mx.array | None) -> mx.array:
            del mask
            return self.activations(ids[None], slots).logits[:, -1, :]

        return compiled_decode(DecodePlan(step=step, promote=promote), cache, capacity)

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
