"""Gemma 3 text (`gemma3_text`): interleaved sliding/full attention, sandwich norms.

Authoritative semantics: transformers' modeling_gemma3.py. What the trunk does that the
Qwen family does not:

- `layer_types` picks, per layer, a sliding-window mask + `rope_local_base_freq` or a
  full causal mask + `rope_theta`. The cache is never evicted; the window lives in the
  mask (a masked key contributes nothing).
- The attention scale is `query_pre_attn_scalar ** -0.5`, which only coincides with
  `head_dim ** -0.5` on this checkpoint. `num_attention_heads * head_dim` (1024) is
  decoupled from `hidden_size` (640).
- Sandwich norms: each residual arm is normed on the way in *and* on the way out.
- Zero-centered RMSNorm: the scale is `1 + w`. Folded on the dict side at load
  (`_fold_norm_scales`), so the tree holds plain `nn.RMSNorm` and no `1 + w` add runs
  per norm per token — 108 extra kernels per step on an 18-layer trunk.
- Embeddings scaled by `sqrt(hidden_size)`; transformers keeps that scalar in float32
  and casts it to the weight dtype, so bf16 sees 25.25 and fp32 25.298221 — the cast is
  reproduced here, not the rounded constant.
- lm_head tied to the embedding table; gelu (tanh approximation) MLP.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU, split_qkv

SLIDING = "sliding_attention"
FULL = "full_attention"

# `hidden_activation` is gelu_pytorch_tanh: the tanh approximation, which is what
# mlx.nn.gelu_approx computes — as a single compiled kernel, hence the reuse instead of
# the expression inline. `mx.compile` erases the signature from the stubs, so it is
# restated here the way `core/mxcompat.py` restates mlx's own stale ones.
if TYPE_CHECKING:

    def _gelu(x: mx.array) -> mx.array: ...

else:
    _gelu = nn.gelu_approx


@dataclass(frozen=True)
class Gemma3TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_local_base_freq: float
    query_pre_attn_scalar: float
    sliding_window: int
    layer_types: tuple[str, ...]
    tie_word_embeddings: bool
    intermediate_size: int
    quantization: tuple[int, int] | None  # (group_size, bits)


def causal_mask(queries: int, keys: int, window: int | None) -> mx.array:
    """Which keys each query may attend to: everything up to itself, and within `window`
    positions of it when the layer slides. The queries are the *last* `queries` of the
    `keys` positions, which is what lets a single query attend to a whole cache."""
    rows = mx.arange(keys - queries, keys).reshape(queries, 1)
    columns = mx.arange(keys).reshape(1, keys)
    within = rows >= columns
    if window is None:
        return within
    return within & (rows < columns + window)


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding = layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self.rope_base = config.rope_local_base_freq if self.sliding else config.rope_theta
        self.scale = config.query_pre_attn_scalar**-0.5
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
            x, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | None:
        """A lone query attends to everything it can reach; only a sliding layer whose
        cache already exceeds the window still needs a mask at T=1."""
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        q, k, v = self.split_heads(x)
        keys, values = cache.update_and_fetch(self.rope(k, offset), v)
        attended = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset),
            keys,
            values,
            scale=self.scale,
            mask=self.mask(length, keys.shape[2]),
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


class Gemma3Block(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_type)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, _gelu)
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cache))
        return attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )


class Gemma3Trunk(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3Block(config, layer_type) for layer_type in config.layer_types]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Gemma3Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Gemma3(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma3Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def embed(self, ids: mx.array) -> mx.array:
        embedded = self.model.embed_tokens(ids)
        scale = mx.array(math.sqrt(self.config.hidden_size), mx.float32).astype(embedded.dtype)
        return embedded * scale

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Gemma3Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.embed(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return Gemma3Activations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits
