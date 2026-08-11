from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.models.gpt_oss.config import GPTOSSConfig
from mlx_omnia.models.gpt_oss.layers.block import GPTOSSBlock
from mlx_omnia.models.gpt_oss.layers.rope import yarn_rope


class GPTOSSTrunk(nn.Module):
    def __init__(self, config: GPTOSSConfig, freqs: mx.array, mscale: float) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GPTOSSBlock(config, freqs, mscale) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GPTOSSActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class GPTOSS(nn.Module):
    def __init__(self, config: GPTOSSConfig) -> None:
        super().__init__()
        self.config = config
        freqs, mscale = yarn_rope(config.head_dim, config.rope_theta, config.rope_scaling)
        mx.eval(freqs)
        self.model = GPTOSSTrunk(config, freqs, mscale)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        """One full cache per layer: the sliding layers keep every key and mask it,
        which is what an evicting cache would have dropped."""
        return [KVCache() for _ in self.model.layers]

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> GPTOSSActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if "sliding_attention" in self.config.layer_types:
            sliding = self._sliding_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            x = block(x, full if kind == "full_attention" else sliding, layer_cache)
            blocks.append(x)
        return GPTOSSActivations(blocks, self.lm_head(self.model.norm(x)))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits

    def _sliding_mask(self, length: int, offset: int) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + window`, built only where it is
        not already something cheaper. No key is old enough for the window to cut while
        `offset + length <= window`, so the band *is* the causal mask there — and at T=1
        the single row is causal by construction, leaving `columns > offset - window`."""
        window = self.config.sliding_window
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)
