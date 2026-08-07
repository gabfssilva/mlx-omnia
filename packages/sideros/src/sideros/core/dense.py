"""The house's standard dense decoder: GQA + RoPE + SwiGLU, pre-norm residuals.

Norm → attention → add → norm → SwiGLU → add, with plain `nn.RMSNorm` and no bias
anywhere. `rotary` is per layer: a False entry attends on content alone (NoPE), which
changes nothing about the cache or the mask.

An architecture that differs in a leaf (per-head q/k norm, post-norm residuals, scalar
multipliers) does not parameterize this tree — it declares its own, as `qwen3.py`,
`olmo2.py` and `granite.py` do.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU, split_qkv
from sideros.core.rope import LlamaRoPEScaling, ScalingJson, llama3_freqs


@dataclass(frozen=True)
class DenseConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: LlamaRoPEScaling | None
    tie_word_embeddings: bool
    intermediate_size: int
    eos_token_id: tuple[int, ...]


class DenseJson(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | list[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    rope_theta: NotRequired[float]
    rope_scaling: NotRequired[ScalingJson | None]
    attention_bias: NotRequired[bool]
    mlp_bias: NotRequired[bool]
    tie_word_embeddings: NotRequired[bool]


class DenseAttention(nn.Module):
    """`rotary=False` is SmolLM3's NoPE layer: the same tree, the rotation skipped. The
    flag lives here because the two architectures share every leaf name."""

    def __init__(self, config: DenseConfig, *, rotary: bool = True) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rotary = rotary
        self._freqs = (
            None
            if config.rope_scaling is None
            else llama3_freqs(config.head_dim, config.rope_theta, config.rope_scaling)
        )
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)

    def rope(self, x: mx.array, offset: int) -> mx.array:
        if not self.rotary:
            return x
        if self._freqs is None:
            return mx.fast.rope(
                x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
            )
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=None, scale=1.0, offset=offset,
            freqs=self._freqs,
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = split_qkv(
            self.qkv_proj(x), heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim
        )
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset), keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


class DenseBlock(nn.Module):
    def __init__(self, config: DenseConfig, *, rotary: bool = True) -> None:
        super().__init__()
        self.self_attn = DenseAttention(config, rotary=rotary)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class DenseTrunk(nn.Module):
    def __init__(self, config: DenseConfig, rotary: tuple[bool, ...] | None = None) -> None:
        super().__init__()
        layers = config.num_hidden_layers
        rotates = rotary if rotary is not None else (True,) * layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DenseBlock(config, rotary=rotates[layer]) for layer in range(layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DenseActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class DenseModel(nn.Module):
    def __init__(self, config: DenseConfig, rotary: tuple[bool, ...] | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = DenseTrunk(config, rotary)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> DenseActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return DenseActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
