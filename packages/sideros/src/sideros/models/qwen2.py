"""Qwen2 dense: Qwen3 minus the per-head q/k RMSNorm and minus explicit head_dim,
plus a bias on q/k/v.

Authoritative semantics: transformers' modeling_qwen2.py. `head_dim` is not in the
config — it is `hidden_size // num_attention_heads` (0.5B: 896 // 14 = 64) — and the
three attention projections carry a bias (`o_proj` does not; the MLP does not). Qwen2
is the only architecture in the house with it.

The qkv load-time fusion is a concatenation on the output axis, and rows are the
fusion axis in every representation the checkpoint uses: dense weight, packed u32
plus scales/biases, and the projection bias vector (whose single axis *is* the output
axis). One fusion rule covers all four.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU, split_qkv


@dataclass(frozen=True)
class Qwen2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    intermediate_size: int
    quantization: tuple[int, int] | None  # (group_size, bits)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        hidden = config.hidden_size
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, hidden + 2 * key_values, bias=True)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """qkv split back into per-head [1, heads, length, head_dim], unrotated — the
        boundary transformers' q_proj/k_proj hooks expose, modulo the head reshape."""
        return split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = self.split_heads(x)
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset), keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


class Qwen2Block(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen2Trunk(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Qwen2(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Qwen2Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return Qwen2Activations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
