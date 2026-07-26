"""GPT-2 as a tree of nn.Module with property names = checkpoint names.

Authoritative semantics: transformers' modeling_gpt2.py. lm_head is tied to wte;
wpe is a learned position table (dense even inside a quantized model).
"""

import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.config import GPT2Config
from sideros.core.cache import KVCache


def _gelu_new(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = (
            part.reshape(batch, length, self.n_head, -1).transpose(0, 2, 1, 3)
            for part in mx.split(qkv, 3, axis=-1)
        )
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        # A lone query attends to everything: no mask on the T=1 step.
        mask = "causal" if length > 1 else None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask=mask
        )
        return self.c_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(_gelu_new(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))


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
        self.h = [Block(config) for _ in range(config.n_layer)]
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
