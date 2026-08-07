from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.gpt2.config import GPT2Config
from sideros.models.gpt2.layers.block import GPT2Block


class GPT2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    ln_f: mx.array
    logits: mx.array


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.h]

    def activations(self, ids: mx.array) -> GPT2Activations:
        x = self.wte(ids) + self.wpe(mx.arange(ids.shape[-1]))
        embeddings = x
        blocks: list[mx.array] = []
        for block in self.h:
            x = block(x)
            blocks.append(x)
        normed = self.ln_f(x)
        return GPT2Activations(embeddings, blocks, normed, self.wte.as_linear(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        if cache is None:
            return self.activations(ids).logits
        offset = cache[0].offset
        x = self.wte(ids) + self.wpe(mx.arange(offset, offset + ids.shape[-1]))
        for block, layer_cache in zip(self.h, cache, strict=True):
            x = block(x, layer_cache)
        return self.wte.as_linear(self.ln_f(x))
