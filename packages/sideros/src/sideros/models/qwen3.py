"""Qwen3 dense: Qwen2 with per-head q/k RMSNorm, no bias, explicit head_dim.

Authoritative semantics: transformers' modeling_qwen3.py. The norms sit between the
projections and the rotation, and `num_attention_heads * head_dim` is decoupled from
`hidden_size` (0.6B: 16x128 = 2048 against a 1024 trunk).

Two load-time fusions, both on the output axis (row-aligned in every representation,
so dense and packed u32 fuse the same way): q‖k‖v into `qkv_proj` and gate‖up into
`mlp.gate_up_proj`. The tree declares only the fused names.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.rope_epilogue import rope_epilogue, rope_epilogue_applies
from sideros.core.layers import SwiGLU, split_qkv


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    intermediate_size: int
    quantization: tuple[int, int] | None  # (group_size, bits)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.eps = config.rms_norm_eps
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """qkv split back into per-head [1, heads, length, head_dim], normed but
        unrotated — the boundary transformers' q_norm/k_norm hooks expose."""
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        return self.q_norm(q), self.k_norm(k), v

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def step_heads(self, x: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        """T=1: q/k norm + rotation in one dispatch, v a free slice of the projection."""
        fused = self.qkv_proj(x)
        queries, keys = rope_epilogue(
            fused,
            query_heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm.weight,
            k_norm=self.k_norm.weight,
            offset=offset,
            base=self.rope_theta,
            eps=self.eps,
        )
        values = fused[..., (self.heads + self.kv_heads) * self.head_dim :]
        return (
            queries.reshape(1, self.heads, 1, self.head_dim),
            keys.reshape(1, self.kv_heads, 1, self.head_dim),
            values.reshape(1, self.kv_heads, 1, self.head_dim),
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        if length == 1 and rope_epilogue_applies(self.head_dim):
            queries, rotated, v = self.step_heads(x, offset)
        else:
            q, k, v = self.split_heads(x)
            queries, rotated = self.rope(q, offset), self.rope(k, offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen3Trunk(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Qwen3Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return Qwen3Activations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
