from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.gpt2.config import GPT2Config
from mlx_omnia.engine.models.gpt2.layers.block import GPT2Block


class GPT2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    ln_f: mx.array
    logits: mx.array


class GPT2(nn.Module):
    continuous_batching = True

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.h]

    def activations(
        self, ids: mx.array, cache: Sequence[KVStore] | None = None
    ) -> GPT2Activations:
        positions = mx.arange(ids.shape[-1])
        if cache is not None:
            offset = cache[0].offset
            # One offset per row under a ragged batch, one for the whole step otherwise.
            start = offset if isinstance(offset, int) else offset[:, None]
            positions = start + positions
        x = self.wte(ids) + self.wpe(positions)
        embeddings = x
        blocks: list[mx.array] = []
        caches: Sequence[KVStore | None] = cache if cache is not None else [None] * len(self.h)
        for block, layer_cache in zip(self.h, caches, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.ln_f(x)
        return GPT2Activations(embeddings, blocks, normed, self.wte.as_linear(normed))

    def __call__(self, ids: mx.array, cache: Sequence[KVStore] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
