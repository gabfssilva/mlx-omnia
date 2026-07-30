"""Gemma 4 text (`gemma4` / `gemma4_unified` / `gemma4_assistant`): PLE, KV sharing,
proportional partial-rotary RoPE, dual head_dim, standard (non-zero-centered) norms,
logit softcap, double-wide MLP on KV-shared layers.

Authoritative semantics: transformers' `modeling_gemma4.py`. Four structures have no
Sideros precedent:

- **Per-Layer Embeddings (PLE)** — a second `[vocab, layers*ple_dim]` embedding table
  scaled by `sqrt(ple_dim)`, a context projection scaled by `1/sqrt(hidden)`, combined
  at `1/sqrt(2)` per layer, and a **third residual arm** per block
  (gate → act → mul-by-ple → project → norm → add → `*= layer_scalar`).
- **KV sharing** — `num_kv_shared_layers` trailing layers own no `k_proj`/`v_proj`/
  `k_norm`/`v_norm`; they read full-length K,V published by the last non-sharing layer
  of the same type. `make_cache` returns a `KVCache` per **non-sharing** layer only;
  the sharing layers get a `SharedKVReader` that references the storing layer's cache.
- **Proportional partial-rotary RoPE** on full-attention layers —
  `partial_rotary_factor=0.25` over `global_head_dim=512`: 64 real frequency pairs
  (128 rotated dims) and 192 zero-frequency pairs (identity). The exponent denominator
  is `global_head_dim`, not the rotated count, so `mx.fast.rope` cannot reproduce it;
  manual RoPE with precomputed fp32 cos/sin is required.
- **Dual head_dim** — sliding layers use `head_dim` (256), full layers use
  `global_head_dim` (512). `scale=1.0` (both Q and K are RMSNormed).

Norms are **standard** RMSNorm (`weight`, no `1+w`) — *not* the Gemma 3 zero-centered
fold. `v_norm` and the MoE router norm are scale-less (`RMSNormNoScale`, no `weight`).
No dict-side qkv fusion (shared layers lack k/v; widths vary by layer type). No
`_fold_norm_scales`. The MLP gate/up stay separate (no dict-side concat) to keep
shared layers clean; the tree declares separate `gate_proj`/`up_proj` as the checkpoint
stores them. Logit softcap `tanh(logits/cap)*cap` runs in fp32. `attention_k_eq_v`
(12B path) drops `v_proj` on full layers (`v = k`).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    drop_tied_head,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.core.cache import KVCache, LayerCache
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

SLIDING = "sliding_attention"
FULL = "full_attention"

# gelu_pytorch_tanh == mlx.nn.gelu_approx — same single-kernel reuse as Gemma 3.
if TYPE_CHECKING:

    def _gelu(x: mx.array) -> mx.array: ...

else:
    _gelu = nn.gelu_approx


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RopeConfig:
    rope_type: str
    theta: float
    partial_rotary_factor: float


@dataclass(frozen=True)
class Gemma4TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_head_dim: int
    vocab_size: int
    sliding_window: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float
    tie_word_embeddings: bool
    final_logit_softcapping: float | None
    hidden_size_per_layer_input: int
    vocab_size_per_layer_input: int
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    attention_k_eq_v: bool
    num_global_key_value_heads: int | None
    enable_moe_block: bool
    sliding_rope: _RopeConfig
    full_rope: _RopeConfig
    eos_token_id: tuple[int, ...]

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def is_kv_shared_layer(self, idx: int) -> bool:
        return self.num_kv_shared_layers > 0 and idx >= self.first_kv_shared_layer_idx

    def store_full_length_kv(self, idx: int) -> bool:
        """The last non-sharing layer of each type publishes full-length K,V."""
        if self.num_kv_shared_layers == 0:
            return False
        first_shared = self.first_kv_shared_layer_idx
        # The layer just before the sharing block of its type stores.
        # For a 4:1 split with 20 shared layers (idx 15..34), the store layers are
        # the last non-sharing layer of each type: idx 13 (sliding) and 14 (full).
        for i in range(first_shared - 1, -1, -1):
            if self.layer_types[i] == self.layer_types[idx]:
                return i == idx
        return False

    def intermediate_for_layer(self, idx: int) -> int:
        if self.use_double_wide_mlp and self.is_kv_shared_layer(idx):
            return self.intermediate_size * 2
        return self.intermediate_size


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #


def causal_mask(queries: int, keys: int, window: int | None) -> mx.array:
    rows = mx.arange(keys - queries, keys).reshape(queries, 1)
    columns = mx.arange(keys).reshape(1, keys)
    within = rows >= columns
    if window is None:
        return within
    return within & (rows < columns + window)


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable weight — for `v_norm` and the MoE router norm.

    Upcast to fp32 for the `_norm`, cast back; same as `nn.RMSNorm` minus the
    multiply-by-weight.
    """

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        ms = mx.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * mx.rsqrt(ms + self.eps)
        return normed.astype(orig_dtype)


# --------------------------------------------------------------------------- #
# Proportional partial-rotary RoPE
# --------------------------------------------------------------------------- #


def proportional_inv_freq(head_dim: int, partial: float, theta: float) -> mx.array:
    """The `inv_freq` vector for proportional partial-rotary RoPE, in fp32.

    `int(partial * head_dim // 2)` real frequency pairs computed with the exponent
    denominator = `head_dim` (not the rotated count), followed by zero-frequency
    pairs (identity: cos=1, sin=0). Matches transformers'
    `_compute_proportional_rope_parameters`.
    """
    rope_angles = int(partial * head_dim // 2)
    real = mx.power(
        theta,
        mx.arange(0, rope_angles, dtype=mx.float32) * (2.0 / head_dim),
    )
    zeros = mx.zeros(head_dim // 2 - rope_angles, dtype=mx.float32)
    return mx.concatenate([real, zeros])


def _cos_sin_tables(
    inv_freq: mx.array, positions: mx.array, head_dim: int
) -> tuple[mx.array, mx.array]:
    """Precompute fp32 cos/sin tables for manual RoPE.

    `positions` is `[length]`; returns `[length, head_dim//2]` each.
    """
    freqs = mx.outer(positions.astype(mx.float32), inv_freq)
    # [length, head_dim//2] — freqs for each position, each frequency pair
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    return cos, sin


def manual_rope(
    x: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    """Apply RoPE manually to `x` of shape `[1, heads, length, head_dim]`.

    The first `2*cos.shape[-1]` dims are rotated as pairs (x_even, x_odd):
    out_even = x_even*cos - x_odd*sin
    out_odd  = x_even*sin + x_odd*cos
    The remaining dims (head_dim - 2*rotated) are identity (cos=1, sin=0 already
    baked by the zero-frequency inv_freq tail, but we slice to avoid relying on
    it — the caller passes cos/sin sized to the rotated count only).
    """
    rotated = cos.shape[-1] * 2
    x_rot = x[..., :rotated]
    x_pass = x[..., rotated:]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    # cos/sin are [length, rotated//2] — broadcast over heads.
    cos_b = cos[mx.newaxis, mx.newaxis, :, :]
    sin_b = sin[mx.newaxis, mx.newaxis, :, :]
    out_even = x_even * cos_b - x_odd * sin_b
    out_odd = x_even * sin_b + x_odd * cos_b
    out_rot = mx.stack([out_even, out_odd], axis=-1).flatten(-2)
    if x_pass.shape[-1] > 0:
        return mx.concatenate([out_rot, x_pass], axis=-1)
    return out_rot


# --------------------------------------------------------------------------- #
# Shared KV cache
# --------------------------------------------------------------------------- #


class SharedKVReader(LayerCache):
    """A cache placeholder for KV-shared layers: no projection, no own buffer.

    Reads the full-length K,V from the storing layer's `KVCache` of the same type.
    `offset` tracks the position so the mask logic works, but the actual K,V come
    from `keys`/`values` set each step by the trunk.
    """

    def __init__(self) -> None:
        super().__init__()
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def is_trimmable(self) -> bool:
        return False

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.layer_type = config.layer_types[layer_idx]
        self.sliding = self.layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self.scale = 1.0

        if self.sliding:
            self.head_dim = config.head_dim
            rope = config.sliding_rope
        else:
            self.head_dim = config.global_head_dim
            rope = config.full_rope
        self.rope_type = rope.rope_type
        self.rope_theta = rope.theta
        self.partial_rotary_factor = rope.partial_rotary_factor
        self.is_full = not self.sliding

        self.k_eq_v = config.attention_k_eq_v and not self.sliding

        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        if self.k_eq_v and config.num_global_key_value_heads is not None:
            self.kv_heads = config.num_global_key_value_heads
        key_values = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, queries, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        kv_shared = config.is_kv_shared_layer(layer_idx)
        self.kv_shared = kv_shared
        if not kv_shared:
            self.k_proj = nn.Linear(hidden, key_values, bias=False)
            if not self.k_eq_v:
                self.v_proj = nn.Linear(hidden, key_values, bias=False)
            self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # Precompute proportional inv_freq for full layers (manual rope).
        if self.is_full and self.rope_type == "proportional":
            self._inv_freq = proportional_inv_freq(
                self.head_dim, self.partial_rotary_factor, self.rope_theta
            )
        else:
            self._inv_freq = None

    def _rope_sliding(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

    def _rope_full(self, x: mx.array, offset: int, length: int) -> mx.array:
        assert self._inv_freq is not None
        positions = mx.arange(offset, offset + length, dtype=mx.int32)
        cos, sin = _cos_sin_tables(self._inv_freq, positions, self.head_dim)
        return manual_rope(x, cos, sin)

    def _apply_rope(self, x: mx.array, offset: int) -> mx.array:
        length = x.shape[2]
        if self.is_full and self.rope_type == "proportional":
            return self._rope_full(x, offset, length)
        return self._rope_sliding(x, offset)

    def mask(self, queries: int, keys: int) -> mx.array | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | SharedKVReader,
    ) -> mx.array:
        length = x.shape[1]

        if self.kv_shared:
            assert isinstance(cache, SharedKVReader)
            q = self.q_proj(x).reshape(
                1, length, self.heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            q = self.q_norm(q)
            q = self._apply_rope(q, cache.offset)
            keys = cache.keys
            values = cache.values
            assert keys is not None and values is not None
            attended = mx.fast.scaled_dot_product_attention(
                q,
                keys,
                values,
                scale=self.scale,
                mask=self.mask(length, keys.shape[2]),
            )
            return self.o_proj(
                attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
            )

        q = self.q_proj(x).reshape(
            1, length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(
            1, length, self.kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        if self.k_eq_v:
            v = k
        else:
            v = self.v_proj(x).reshape(
                1, length, self.kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q = self._apply_rope(q, cache.offset)
        k = self._apply_rope(k, cache.offset)

        assert isinstance(cache, KVCache)
        keys, values = cache.update_and_fetch(k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q,
            keys,
            values,
            scale=self.scale,
            mask=self.mask(length, keys.shape[2]),
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.intermediate_for_layer(layer_idx)
        self.inner = inner
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_gelu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# PLE residual arm
# --------------------------------------------------------------------------- #


class PLEArm(nn.Module):
    """The third residual arm: gate → act → mul-by-ple → project → norm → add."""

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        ple_dim = config.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(hidden, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, hidden, bias=False)
        self.post_per_layer_input_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, per_layer_input: mx.array) -> mx.array:
        gate = _gelu(self.per_layer_input_gate(x))
        ple = gate * per_layer_input
        projected = self.per_layer_projection(ple)
        return self.post_per_layer_input_norm(projected)


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #


class Gemma4Block(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.layer_scalar = mx.ones((1,), dtype=mx.float32)
        self.has_ple = config.hidden_size_per_layer_input > 0
        if self.has_ple:
            self.ple_arm = PLEArm(config)

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | SharedKVReader,
        per_layer_input: mx.array | None = None,
    ) -> mx.array:
        attended = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cache)
        )
        h = attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )
        if self.has_ple:
            assert per_layer_input is not None
            h = h + self.ple_arm(h, per_layer_input)
        return h * self.layer_scalar


# --------------------------------------------------------------------------- #
# Trunk + top
# --------------------------------------------------------------------------- #


class Gemma4Trunk(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma4Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

        if config.hidden_size_per_layer_input > 0:
            ple_dim = config.hidden_size_per_layer_input
            n = config.num_hidden_layers
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input, n * ple_dim
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size, n * ple_dim, bias=False
            )
            self.per_layer_projection_norm = nn.RMSNorm(ple_dim, eps=config.rms_norm_eps)


class Gemma4Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Gemma4(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | SharedKVReader]:
        cache: list[KVCache | SharedKVReader] = []
        for i in range(self.config.num_hidden_layers):
            if self.config.is_kv_shared_layer(i):
                cache.append(SharedKVReader())
            else:
                cache.append(KVCache())
        return cache

    def embed(self, ids: mx.array) -> mx.array:
        embedded = self.model.embed_tokens(ids)
        # sqrt(hidden_size) — 1536 is NOT a perfect square, so reproduce the bf16 cast.
        scale = mx.array(math.sqrt(self.config.hidden_size), mx.float32).astype(embedded.dtype)
        return embedded * scale

    def _per_layer_inputs(self, ids: mx.array) -> mx.array | None:
        """The PLE pipeline: token identity + context projection, combined at 1/sqrt(2).

        The context projection takes the **scaled** main embedding (h = embed * sqrt(hidden)),
        matching mlx-lm's `_project_per_layer_inputs(h, ...)`.
        """
        cfg = self.config
        if cfg.hidden_size_per_layer_input == 0:
            return None
        ple_dim = cfg.hidden_size_per_layer_input
        n = cfg.num_hidden_layers

        # Token-identity: scaled embedding lookup, reshaped to [1, S, n, ple_dim].
        ple_embed = self.model.embed_tokens_per_layer(ids)
        ple_scale = mx.array(math.sqrt(ple_dim), mx.float32).astype(ple_embed.dtype)
        token_identity = (ple_embed * ple_scale).reshape(
            1, ple_embed.shape[1], n, ple_dim
        )

        # Context: project the SCALED main embedding → n*ple_dim, scale by 1/sqrt(hidden), norm.
        main_embed = self.model.embed_tokens(ids)
        h = main_embed * mx.array(
            math.sqrt(cfg.hidden_size), mx.float32
        ).astype(main_embed.dtype)
        ctx = self.model.per_layer_model_projection(h)
        ctx_scale = mx.array(
            1.0 / math.sqrt(cfg.hidden_size), mx.float32
        ).astype(ctx.dtype)
        ctx = (ctx * ctx_scale).reshape(1, ctx.shape[1], n, ple_dim)
        ctx = self.model.per_layer_projection_norm(ctx)

        # Combine.
        combine = mx.array(
            1.0 / math.sqrt(2.0), mx.float32
        ).astype(token_identity.dtype)
        return (token_identity + ctx) * combine

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def _softcap(self, logits: mx.array) -> mx.array:
        cap = self.config.final_logit_softcapping
        if cap is None:
            return logits
        logits32 = logits.astype(mx.float32)
        cap_arr = mx.array(cap, mx.float32)
        return (mx.tanh(logits32 / cap_arr) * cap_arr).astype(logits.dtype)

    def activations(
        self,
        ids: mx.array,
        cache: list[KVCache | SharedKVReader] | None = None,
    ) -> Gemma4Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.embed(ids)
        embeddings = x
        per_layer_input = self._per_layer_inputs(ids)
        blocks: list[mx.array] = []

        # Track shared KV: for each layer_type, the KVCache and pre-update offset
        # of the storing layer. The offset is captured BEFORE the store layer runs,
        # so the shared layer's q is rotated by the correct starting position.
        shared_kv: dict[str, tuple[KVCache, int]] = {}

        for i, (block, layer_cache) in enumerate(
            zip(self.model.layers, cache, strict=True)
        ):
            # If this is a storing layer, capture its pre-update offset for shared layers.
            if self.config.store_full_length_kv(i):
                assert isinstance(layer_cache, KVCache)
                shared_kv[self.config.layer_types[i]] = (layer_cache, layer_cache.offset)

            # If this is a shared layer, publish full-length KV from the store.
            # The keys/values are the store's post-update cache (includes current step);
            # the offset is the store's pre-update offset (the q starting position).
            if self.config.is_kv_shared_layer(i):
                store_cache, store_offset = shared_kv[self.config.layer_types[i]]
                assert isinstance(store_cache, KVCache)
                assert isinstance(layer_cache, SharedKVReader)
                layer_cache.keys, layer_cache.values = store_cache.fetch()
                layer_cache.offset = store_offset

            # Extract per-layer slice for this block.
            ple_slice = None
            if per_layer_input is not None:
                ple_slice = per_layer_input[:, :, i, :]

            x = block(x, layer_cache, ple_slice)
            blocks.append(x)

        normed = self.model.norm(x)
        logits = self._softcap(self.head(normed))
        return Gemma4Activations(embeddings, blocks, normed, logits)

    def __call__(
        self, ids: mx.array, cache: list[KVCache | SharedKVReader] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits


# --------------------------------------------------------------------------- #
# Config parsing + loader
# --------------------------------------------------------------------------- #


class _RopeJson(TypedDict):
    rope_type: str
    rope_theta: NotRequired[float]
    partial_rotary_factor: NotRequired[float]


class _TextJson(TypedDict):
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_head_dim: NotRequired[int]
    vocab_size: int
    sliding_window: int
    layer_types: NotRequired[list[str]]
    rms_norm_eps: float
    tie_word_embeddings: NotRequired[bool]
    final_logit_softcapping: NotRequired[float | None]
    hidden_size_per_layer_input: NotRequired[int]
    vocab_size_per_layer_input: NotRequired[int]
    num_kv_shared_layers: NotRequired[int]
    use_double_wide_mlp: NotRequired[bool]
    attention_k_eq_v: NotRequired[bool]
    num_global_key_value_heads: NotRequired[int]
    enable_moe_block: NotRequired[bool]
    rope_parameters: NotRequired[dict[str, _RopeJson]]
    eos_token_id: NotRequired[int | list[int]]
    query_pre_attn_scalar: NotRequired[float]


class _Json(TypedDict):
    model_type: str
    text_config: NotRequired[_TextJson]
    hidden_size: NotRequired[int]
    intermediate_size: NotRequired[int]
    num_hidden_layers: NotRequired[int]
    num_attention_heads: NotRequired[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    global_head_dim: NotRequired[int]
    vocab_size: NotRequired[int]
    sliding_window: NotRequired[int]
    layer_types: NotRequired[list[str]]
    rms_norm_eps: NotRequired[float]
    tie_word_embeddings: NotRequired[bool]
    final_logit_softcapping: NotRequired[float | None]
    hidden_size_per_layer_input: NotRequired[int]
    vocab_size_per_layer_input: NotRequired[int]
    num_kv_shared_layers: NotRequired[int]
    use_double_wide_mlp: NotRequired[bool]
    attention_k_eq_v: NotRequired[bool]
    num_global_key_value_heads: NotRequired[int]
    enable_moe_block: NotRequired[bool]
    rope_parameters: NotRequired[dict[str, _RopeJson]]
    eos_token_id: NotRequired[int | list[int]]


def _rope_config(raw: _RopeJson, default_theta: float) -> _RopeConfig:
    rope_type = raw.get("rope_type", "default")
    theta = raw.get("rope_theta", default_theta)
    partial = raw.get("partial_rotary_factor", 1.0)
    return _RopeConfig(rope_type=rope_type, theta=theta, partial_rotary_factor=partial)


def _build_layer_types(n: int) -> tuple[str, ...]:
    """4 sliding : 1 full, same as Gemma 3 / Gemma 4 E2B."""
    return tuple(SLIDING if i % 5 != 4 else FULL for i in range(n))


def _config(path: Path) -> Gemma4TextConfig:
    raw: _Json = json.loads(path.read_text())
    model_type = raw["model_type"]
    if model_type not in ("gemma4", "gemma4_unified", "gemma4_assistant", "gemma4_text"):
        raise ValueError(f"expected gemma4* model_type, got {model_type!r}")

    # Standalone text config or multimodal wrapper with text_config.
    text: _TextJson = raw.get("text_config", raw)  # type: ignore[assignment]

    global_head_dim = text.get("global_head_dim", text["head_dim"])
    n_layers = text["num_hidden_layers"]
    layer_types = text.get("layer_types")
    if layer_types is None:
        layer_types = list(_build_layer_types(n_layers))
    unknown = set(layer_types) - {SLIDING, FULL}
    if unknown:
        raise ValueError(f"unknown layer types {sorted(unknown)}")

    rope_params = text.get("rope_parameters", {})
    sliding_rope = _rope_config(
        rope_params.get(
            "sliding_attention",
            {"rope_type": "default", "rope_theta": 1e4, "partial_rotary_factor": 1.0},
        ),
        1e4,
    )
    full_rope = _rope_config(
        rope_params.get(
            "full_attention",
            {"rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.25},
        ),
        1e6,
    )

    eos = text.get("eos_token_id", raw.get("eos_token_id", 0))
    tied = text.get("tie_word_embeddings", raw.get("tie_word_embeddings", True))

    return Gemma4TextConfig(
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        global_head_dim=global_head_dim,
        vocab_size=text["vocab_size"],
        sliding_window=text["sliding_window"],
        layer_types=tuple(layer_types),
        rms_norm_eps=text["rms_norm_eps"],
        tie_word_embeddings=tied,
        final_logit_softcapping=text.get("final_logit_softcapping"),
        hidden_size_per_layer_input=text.get("hidden_size_per_layer_input", 0),
        vocab_size_per_layer_input=text.get(
            "vocab_size_per_layer_input", text["vocab_size"]
        ),
        num_kv_shared_layers=text.get("num_kv_shared_layers", 0),
        use_double_wide_mlp=text.get("use_double_wide_mlp", False),
        attention_k_eq_v=text.get("attention_k_eq_v", False),
        num_global_key_value_heads=text.get("num_global_key_value_heads"),
        enable_moe_block=text.get("enable_moe_block", False),
        sliding_rope=sliding_rope,
        full_rope=full_rope,
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _renamed(name: str) -> str | None:
    """Extract the text tower from a multimodal checkpoint.

    Raw HF `Gemma4ForConditionalGeneration`: `model.language_model.*` for the text
    tower, `model.vision_tower.*` / `model.audio_*` for the modality towers. We keep
    only `model.language_model.*` → `model.*` and discard everything else.
    Standalone text checkpoints already use `model.*` and pass through unchanged.
    """
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    # Standalone text checkpoints: model.* (but NOT model.vision_tower.*, etc.).
    if name.startswith("model."):
        suffix = name.removeprefix("model.")
        if suffix.startswith((
            "vision_tower", "multi_modal_projector", "audio_tower",
            "embed_audio", "embed_vision", "vision_embedder", "visual",
        )):
            return None
        return name
    # lm_head at top level (untied) stays.
    if name.startswith("lm_head."):
        return name
    # Everything else (vision_tower, audio, multimodal_projector, etc.) is discarded.
    return None


def _drop_unused_attn_weights(
    weights: dict[str, mx.array], config: Gemma4TextConfig
) -> None:
    """Drop attention weights the tree does not declare, so strict load stays exact.

    - KV-shared layers (idx >= first_kv_shared): no k_proj/v_proj/k_norm/v_norm.
    - k_eq_v full layers: no v_proj (v = k).
    """
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.self_attn."
        is_shared = config.is_kv_shared_layer(layer_idx)
        is_k_eq_v_full = (
            config.attention_k_eq_v
            and config.layer_types[layer_idx] == FULL
        )
        drop = set[str]()
        if is_shared:
            drop.update(("k_proj", "v_proj", "k_norm", "v_norm"))
        elif is_k_eq_v_full:
            drop.add("v_proj")
        for name in drop:
            for suffix in ("weight", "scales", "biases"):
                weights.pop(f"{prefix}{name}.{suffix}", None)


def _weights(
    directory: Path, config: Gemma4TextConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    loaded: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        reject_dtype_cast(dtype, part)
        for name, array in part.items():
            renamed = _renamed(name)
            if renamed is None:
                continue
            loaded[renamed] = array.astype(dtype) if dtype is not None else array

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    _drop_unused_attn_weights(loaded, config)

    # No qkv fusion, no gate_up concat, no _fold_norm_scales.
    return loaded


def _composite(directory: Path, model: Gemma4) -> LanguageModel[ModelInput]:
    from sideros.models.gemma3 import Gemma3Tokenizer

    return CompositeModel(
        TextLanguageModel(
            model,
            Gemma3Tokenizer.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    _config,
    Gemma4,
    _weights,
    _composite,
)
