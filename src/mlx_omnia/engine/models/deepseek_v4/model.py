from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.models.deepseek_v4.config import OVERLAP, DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.block import DeepseekV4Trunk
from mlx_omnia.engine.models.deepseek_v4.layers.cache import (
    BatchedDeepseekV4Cache,
    DeepseekV4Cache,
)

type V4Cache = DeepseekV4Cache | BatchedDeepseekV4Cache


class DeepseekV4Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class DeepseekV4(nn.Module):
    continuous_batching = True

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
        self, ids: mx.array, cache: Sequence[V4Cache] | None = None
    ) -> DeepseekV4Activations:
        cache = cache if cache is not None else self.make_cache()
        if isinstance(cache[0], BatchedDeepseekV4Cache):
            return self._ragged(ids, cache)
        single: list[DeepseekV4Cache] = []
        for layer in cache:
            if not isinstance(layer, DeepseekV4Cache):
                raise TypeError(f"a deepseek_v4 forward mixes in {type(layer).__name__}")
            single.append(layer)
        return self._forward(ids, single)

    def _ragged(self, ids: mx.array, cache: Sequence[V4Cache]) -> DeepseekV4Activations:
        """A ragged batch, one row at a time.

        Every scalar this family carries per sequence is genuinely per row — the
        compressor's remainder and partial tail, the pooled length, the sparse selection the
        indexer makes, the position the local window is cut at — and the trunk below the
        attention (the fused mHC junction, the MoE's single-token kernels) is written for one
        row. So the batch is decomposed here and each row runs the forward it already ran
        alone, which is bit-identical by construction.
        """
        layers: list[BatchedDeepseekV4Cache] = []
        for layer in cache:
            if not isinstance(layer, BatchedDeepseekV4Cache):
                raise TypeError("a batched deepseek_v4 forward mixes in a solo layer")
            layers.append(layer)
        count = len(layers[0].rows)
        parts = [
            self._forward(ids[row : row + 1], [layer.rows[row] for layer in layers])
            for row in range(count)
        ]
        blocks = [
            mx.concatenate(list(pieces))
            for pieces in zip(*(part.blocks for part in parts), strict=True)
        ]
        return DeepseekV4Activations(blocks, mx.concatenate([part.logits for part in parts]))

    def _forward(
        self, ids: mx.array, cache: Sequence[DeepseekV4Cache]
    ) -> DeepseekV4Activations:
        x = self.model.embed_tokens(ids)
        mask = self._window(x.shape[1], cache[0].offset)
        h = mx.contiguous(
            mx.broadcast_to(
                x[:, :, None, :], (x.shape[0], x.shape[1], self.config.hc_mult, x.shape[2])
            )
        )
        blocks: list[mx.array] = []
        layers = self.model.layers
        partials: mx.array | None = None
        for index, (block, layer_cache) in enumerate(zip(layers, cache, strict=True)):
            following = layers[index + 1] if index + 1 < len(layers) else None
            next_fn = (
                following.attn_hc.fn
                if following is not None and following.attn_hc.fused
                else None
            )
            h, partials = block(h, mask, layer_cache, ids, partials, next_fn)
            blocks.append(h)
        normed = self.model.norm(self.model.hc_head(h))
        return DeepseekV4Activations(blocks, self.lm_head(normed))

    def __call__(self, ids: mx.array, cache: Sequence[V4Cache] | None = None) -> mx.array:
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
