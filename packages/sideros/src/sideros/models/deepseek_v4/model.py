from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.models.deepseek_v4.config import OVERLAP, DeepseekV4Config
from sideros.models.deepseek_v4.layers.block import DeepseekV4Trunk
from sideros.models.deepseek_v4.layers.cache import DeepseekV4Cache


class DeepseekV4Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class DeepseekV4(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        self.model = DeepseekV4Trunk(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[DeepseekV4Cache]:
        """One cache per layer, shaped by that layer's compress ratio: the pooled halves
        only exist where the layer has a compressor."""
        return [
            DeepseekV4Cache(ratio, indexed=ratio == OVERLAP) for ratio in self.config.ratios
        ]

    def activations(
        self, ids: mx.array, cache: list[DeepseekV4Cache] | None = None
    ) -> DeepseekV4Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        mask = self._window(x.shape[1], cache[0].offset)
        h = mx.contiguous(
            mx.broadcast_to(
                x[:, :, None, :], (x.shape[0], x.shape[1], self.config.hc_mult, x.shape[2])
            )
        )
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            h = block(h, mask, layer_cache, ids)
            blocks.append(h)
        normed = self.model.norm(self.model.hc_head(h))
        return DeepseekV4Activations(blocks, self.lm_head(normed))

    def __call__(self, ids: mx.array, cache: list[DeepseekV4Cache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits

    def _window(self, length: int, offset: int) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + 128`, built only where it is not
        already something cheaper: no key is old enough for the window to cut while
        `offset + length <= window`."""
        window = self.config.sliding_window
        total = offset + length
        if total <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(total).reshape(1, -1)
        rows = mx.arange(offset, total).reshape(-1, 1)
        return (rows >= columns) & (rows < columns + window)
