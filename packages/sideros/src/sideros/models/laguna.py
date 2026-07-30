"""Laguna S 2.1 (`laguna`): interleaved SWA/full attention with per-layer head counts,
softplus attention gating, sigmoid-routed MoE with a per-expert selection bias and a
shared expert.

Authoritative semantics: transformers' `modeling_laguna.py`. What the trunk does that
no previous ported model has all at once:

- **Per-layer head count** (`num_attention_heads_per_layer`): 48 query heads on full
  layers, 72 on sliding. KV heads are constant at 8, so the GQA ratio varies by layer.
- **Per-layer-type RoPE**: full layers use YaRN (factor 128, original 8192) with
  `partial_rotary_factor` 0.5 — the first 64 of 128 dims rotate; the rest pass through.
  Sliding layers use default RoPE (theta 10k) with full rotary. The YaRN `mscale`
  (attention_factor 1.4852) pre-scales only the rotary dims, not the pass-through.
- **Softplus attention gating** (per-head): `softplus(g_proj(x))` in float32 scales each
  head's output before `o_proj`.
- **Sigmoid routing** (not softmax): scores are independent; selection adds a
  `e_score_correction_bias`; weights come from the unbiased sigmoid scores and
  renormalize. A `routed_scaling_factor` (2.5) scales the routed output before the
  shared expert is added. `softmax_topk` does NOT apply — routing is plain ops.

qkv is fused on the output axis, gate‖up row-interleaved for the decode kernel, and
experts stacked `[E, out, in]` via `SwitchLinear` — all at load, dict-side.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.core.cache import KVCache
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import sigmoid_topk, softmax_topk_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchLinear,
    sorted_gather,
    split_qkv,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class LagunaYaRNScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    attention_factor: float


@dataclass(frozen=True)
class LagunaRopeParams:
    rope_theta: float
    partial_rotary_factor: float
    yarn: LagunaYaRNScaling | None


@dataclass(frozen=True)
class LagunaConfig:
    hidden_size: int
    num_hidden_layers: int
    head_dim: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    sliding_window: int
    tie_word_embeddings: bool
    intermediate_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    moe_router_logit_softcapping: float
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    num_attention_heads_per_layer: tuple[int, ...]
    full_rope: LagunaRopeParams
    sliding_rope: LagunaRopeParams
    eos_token_id: tuple[int, ...]


def _yarn_freqs(
    rotary_dim: int, base: float, scaling: LagunaYaRNScaling
) -> tuple[mx.array, float]:
    """NTK-by-parts frequency table and the mscale that pre-scales q/k before the
    rotation. Same formula as the gpt_oss ``yarn_rope`` and mlx-vlm's ``YarnRoPE``."""
    factor = scaling.factor
    original = scaling.original_max_position_embeddings

    def correction(rotations: float) -> float:
        return (rotary_dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction(scaling.beta_fast)), 0)
    high = min(math.ceil(correction(scaling.beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001

    extra = base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
    inter = factor * extra
    ramp = mx.clip(
        (mx.arange(rotary_dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1
    )
    mask = 1.0 - ramp
    freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
    return freqs, scaling.attention_factor


class SwitchGLU(nn.Module):
    """Gate and up fused row-interleaved ([g0,u0,g1,u1,…) at load: one gather reads both."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]

    def __call__(
        self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class LagunaSparseMoe(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.shared_expert = SwiGLU(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.routed_scaling_factor
        self.cap = config.moe_router_logit_softcapping

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Sigmoid routing (not softmax): scores are independent, selection adds the
        bias, weights come from the unbiased scores and renormalize."""
        logits = self.gate(x).astype(mx.float32)
        if self.cap > 0.0:
            logits = mx.tanh(logits / self.cap) * self.cap
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias.astype(scores.dtype)
        chosen = mx.argpartition(biased, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights.astype(x.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        routed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        return routed * self.scaling + self.shared_expert(x)


class LagunaAttention(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads_per_layer[layer_idx]
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_type = config.layer_types[layer_idx]
        self.sliding = layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        rope = config.sliding_rope if self.sliding else config.full_rope
        self._rotary_dim = int(self.head_dim * rope.partial_rotary_factor)
        if rope.yarn is not None:
            self._freqs, self._mscale = _yarn_freqs(
                self._rotary_dim, rope.rope_theta, rope.yarn
            )
            mx.eval(self._freqs)
        else:
            self._freqs = None
            self._mscale = 1.0
            self._base = rope.rope_theta

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        if self._freqs is not None:
            if self._mscale != 1.0:
                x = mx.concatenate(
                    [
                        x[..., : self._rotary_dim] * self._mscale,
                        x[..., self._rotary_dim :],
                    ],
                    axis=-1,
                )
            return mx.fast.rope(
                x,
                self._rotary_dim,
                traditional=False,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=self._freqs,
            )
        return mx.fast.rope(
            x, self._rotary_dim, traditional=False, base=self._base, scale=1.0, offset=offset
        )

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        queries = self._rope(self.q_norm(q), offset)
        rotated = self._rope(self.k_norm(k), offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, query_width)
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = (
            output.reshape(1, length, self.heads, self.head_dim) * gate[..., None]
        ).reshape(1, length, query_width)
        return self.o_proj(output)


class LagunaBlock(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LagunaAttention(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = LagunaSparseMoe(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _fused_step_applies(self) -> bool:
        mlp = self.mlp
        if not isinstance(mlp, LagunaSparseMoe):
            return False
        gate_up = mlp.switch_mlp.gate_up_proj
        down = mlp.switch_mlp.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            # The gemv kernels read an affine bias per group; MXFP carries none.
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                mlp.hidden, mlp.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            # The routing kernel skips the softcap; a capped config takes the op chain.
            and mlp.cap == 0.0
            and softmax_topk_applies(mlp.split + mlp.k, mlp.k)
        )

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and self._fused_step_applies():
            mlp = self.mlp
            assert isinstance(mlp, LagunaSparseMoe)
            gate_up = mlp.switch_mlp.gate_up_proj
            down = mlp.switch_mlp.down_proj
            assert isinstance(gate_up, QuantizedSwitchLinear)
            assert isinstance(down, QuantizedSwitchLinear)
            assert gate_up.biases is not None and down.biases is not None
            chosen, weights = sigmoid_topk(
                mlp.gate(h).reshape(-1),
                mlp.e_score_correction_bias,
                mlp.k,
                scale=mlp.scaling,
            )
            act = moe_gate_up_act(
                h.reshape(-1),
                gate_up.weight,
                gate_up.scales,
                gate_up.biases,
                chosen,
                group_size=gate_up.group_size,
                bits=gate_up.bits,
            )
            # The shared expert quantizes at different bits than the routed stack, so
            # it can't ride the kernel's shared slot; it folds into the residual input
            # instead, and routed_scaling folds into the routing weights.
            residual = attended + mlp.shared_expert(h)
            return moe_down_combine(
                act.reshape(-1),
                down.weight,
                down.scales,
                down.biases,
                chosen,
                weights,
                residual.reshape(-1),
                group_size=down.group_size,
                bits=down.bits,
            ).reshape(1, 1, mlp.hidden)
        return attended + self.mlp(h)


class LagunaTrunk(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LagunaBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LagunaActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Laguna(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LagunaTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def _sliding_mask(self, length: int, offset: int) -> mx.array | str | None:
        """The band `rows >= columns and rows < columns + window`, built only where
        it is not already something cheaper. No key is old enough for the window to
        cut while `offset + length <= window`, so the band *is* the causal mask there
        — and at T=1 the single row is causal by construction, leaving `columns >
        offset - window`."""
        window = self.config.sliding_window
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)

    def activations(
        self, ids: mx.array, cache: list[KVCache] | None = None
    ) -> LagunaActivations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if SLIDING in self.config.layer_types:
            sliding = self._sliding_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            x = block(x, full if kind == FULL else sliding, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LagunaActivations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


class _YarnRopeJson(TypedDict):
    rope_theta: float
    rope_type: str
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    attention_factor: float
    partial_rotary_factor: float


class _DefaultRopeJson(TypedDict):
    rope_theta: float
    rope_type: str
    partial_rotary_factor: float


class _RopeParametersJson(TypedDict):
    full_attention: _YarnRopeJson
    sliding_attention: _DefaultRopeJson


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    head_dim: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    sliding_window: int
    tie_word_embeddings: bool
    intermediate_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_routed_scaling_factor: float
    moe_router_logit_softcapping: float
    layer_types: list[str]
    mlp_layer_types: list[str]
    num_attention_heads_per_layer: list[int]
    rope_parameters: _RopeParametersJson
    eos_token_id: int | list[int]


def _config(path: Path) -> LagunaConfig:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "laguna":
        raise ValueError(f"expected model_type laguna, got {raw['model_type']!r}")
    unknown = set(raw["layer_types"]) - {FULL, SLIDING}
    if unknown:
        raise ValueError(f"unknown layer types {sorted(unknown)}")
    full_rope_raw = raw["rope_parameters"]["full_attention"]
    sliding_rope_raw = raw["rope_parameters"]["sliding_attention"]
    eos = raw["eos_token_id"]
    return LagunaConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        head_dim=raw["head_dim"],
        num_key_value_heads=raw["num_key_value_heads"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        sliding_window=raw["sliding_window"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        intermediate_size=raw["intermediate_size"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        shared_expert_intermediate_size=raw["shared_expert_intermediate_size"],
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        routed_scaling_factor=raw["moe_routed_scaling_factor"],
        moe_router_logit_softcapping=raw["moe_router_logit_softcapping"],
        layer_types=tuple(raw["layer_types"]),
        mlp_layer_types=tuple(raw["mlp_layer_types"]),
        num_attention_heads_per_layer=tuple(raw["num_attention_heads_per_layer"]),
        full_rope=LagunaRopeParams(
            rope_theta=full_rope_raw["rope_theta"],
            partial_rotary_factor=full_rope_raw["partial_rotary_factor"],
            yarn=LagunaYaRNScaling(
                factor=full_rope_raw["factor"],
                original_max_position_embeddings=full_rope_raw[
                    "original_max_position_embeddings"
                ],
                beta_fast=full_rope_raw["beta_fast"],
                beta_slow=full_rope_raw["beta_slow"],
                attention_factor=full_rope_raw["attention_factor"],
            ),
        ),
        sliding_rope=LagunaRopeParams(
            rope_theta=sliding_rope_raw["rope_theta"],
            partial_rotary_factor=sliding_rope_raw["partial_rotary_factor"],
            yarn=None,
        ),
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _stack_experts(weights: dict[str, mx.array], config: LagunaConfig) -> dict[str, mx.array]:
    """Per-expert gate/up/down stacked into SwitchLinear leaves, gate‖up
    row-interleaved; e_score_correction_bias moves from experts.* to mlp.*."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp."

        bias_key = f"{prefix}experts.e_score_correction_bias"
        if bias_key in weights:
            weights[f"{prefix}e_score_correction_bias"] = weights.pop(bias_key)

        for suffix in ("weight", "scales", "biases"):
            gate_keys = [
                f"{prefix}experts.{e}.gate_proj.{suffix}"
                for e in range(config.num_experts)
            ]
            up_keys = [
                f"{prefix}experts.{e}.up_proj.{suffix}"
                for e in range(config.num_experts)
            ]
            if not all(key in weights for key in gate_keys + up_keys):
                continue
            gates = mx.stack([weights.pop(key) for key in gate_keys])
            ups = mx.stack([weights.pop(key) for key in up_keys])
            interleaved = mx.stack([gates, ups], axis=2).reshape(
                config.num_experts, 2 * config.moe_intermediate_size, -1
            )
            mx.eval(interleaved)
            weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = interleaved

        for suffix in ("weight", "scales", "biases"):
            down_keys = [
                f"{prefix}experts.{e}.down_proj.{suffix}"
                for e in range(config.num_experts)
            ]
            if not all(key in weights for key in down_keys):
                continue
            stacked = mx.stack([weights.pop(key) for key in down_keys])
            mx.eval(stacked)
            weights[f"{prefix}switch_mlp.down_proj.{suffix}"] = stacked

    return weights


def _rename_vlm(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """The mlx-vlm conversion nests the trunk under `language_model.` and the router
    under `gate.proj`, with the selection bias as the router's sibling. The tree keeps
    the original checkpoint's names; pure renames, no arithmetic."""
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        key = key.removeprefix("language_model.")
        key = key.replace(".mlp.gate.proj.", ".mlp.gate.")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
        renamed[key] = value
    return renamed


def _interleave_stacked_experts(
    weights: dict[str, mx.array], config: LagunaConfig
) -> dict[str, mx.array]:
    """A conversion that already stacked the experts ships `switch_mlp.gate_proj` and
    `switch_mlp.up_proj` as `[E, inner, ·]`; only the row interleave is left. Same row
    rule as `_stack_experts`, holding for packed weight, scales and biases."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp.switch_mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            gates, ups = (weights.pop(key) for key in keys)
            interleaved = mx.stack([gates, ups], axis=2).reshape(
                config.num_experts, 2 * config.moe_intermediate_size, -1
            )
            mx.eval(interleaved)
            weights[f"{prefix}gate_up_proj.{suffix}"] = interleaved
    return weights


def _fuse_shared(weights: dict[str, mx.array], config: LagunaConfig) -> dict[str, mx.array]:
    """Shared expert's gate‖up concatenated on the output axis, like the dense MLP."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp.shared_expert."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _weights(directory: Path, config: LagunaConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    """e_score_correction_bias ships float32 in a bfloat16 checkpoint and stays
    float32: the router adds it in float32 whatever the model's precision."""
    weights = _rename_vlm(load_shards(directory))
    reject_dtype_cast(dtype, weights)

    if dtype is not None:
        weights = {
            key: value
            if key.endswith("e_score_correction_bias")
            else value.astype(dtype)
            for key, value in weights.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(weights)

    weights = fuse_qkv(weights, config.num_hidden_layers)
    weights = concat_gate_up(weights, config.num_hidden_layers)
    weights = _stack_experts(weights, config)
    weights = _interleave_stacked_experts(weights, config)
    return _fuse_shared(weights, config)


def _composite(directory: Path, model: Laguna) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint((
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ), _config, Laguna, _weights, _composite)
