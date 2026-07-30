"""Hunyuan 3 (Hy3, `hy_v3`): a 80-layer sigmoid-MoE trunk with Qwen3-style attention.

Authoritative semantics: transformers' `modeling_hy_v3.py`. What the trunk does:

- **Dense layer 0, sparse layers 1-79** (`mlp_layer_types = ["dense"] + ["sparse"]*79`).
  Layer 0 is a `SwiGLU(4096, 13312)`; layers 1-79 are sigmoid-routed MoE with 192
  experts, 8 per token, plus 1 shared expert (width 1536 = `moe_intermediate_size`).
- **Sigmoid routing** (DeepSeek-V3 family, same as Laguna): `sigmoid(logits)` →
  `+ e_score_correction_bias` (selection only) → `topk(k=8)` → weights from the
  **unbiased** sigmoid scores → renorm → `x router_scaling_factor (2.826)`. The router
  gemv runs in **fp32** (`F.linear(x.float(), w.float())` in transformers), and
  `e_score_correction_bias` is fp32 — both kept fp32 through the dtype cast.
- **Shared expert added raw** (`routed + shared`), no gating. The checkpoint sets
  `enable_moe_fp32_combine: false`, so the sum runs in the model dtype.
- **Attention**: GQA (64/8 heads, head_dim 128), qk-norm **before** RoPE, default RoPE
  (theta 11_158_840) on full head_dim. No sliding window, no sinks. This is Qwen3
  attention.
- **MTP head** (layer 80) is dropped by transformers (`_keys_to_ignore_on_load_unexpected
  = [r"model\\.layers\\.80.*"]`). Inference is 1-token/step AR.

qkv is fused on the output axis, gate‖up row-interleaved for the decode kernel, experts
stacked `[E, out, in]` via `SwitchLinear` — all at load, dict-side. The router weight
(`mlp.gate.weight`) and `e_score_correction_bias` stay fp32 across the dtype cast.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

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
from sideros.core.kernels.add_rms_norm import add_rms_norm, add_rms_norm_applies
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

# Default off (see Qwen3-MoE): rope_epilogue and add_rms_norm each lose decode on the
# 30B step. Kept wired for re-measurement on Hy3 once a local checkpoint exists.
ROPE_EPILOGUE_KERNEL = False
ADD_RMS_NORM_KERNEL = False


@dataclass(frozen=True)
class Hy3Config:
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
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    router_scaling_factor: float
    enable_moe_fp32_combine: bool
    mlp_layer_types: tuple[str, ...]
    eos_token_id: tuple[int, ...]


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


class Hy3SparseMoe(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
        self.shared_expert = SwiGLU(config.hidden_size, shared_intermediate)
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.router_scaling_factor
        self.fp32_combine = config.enable_moe_fp32_combine

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Sigmoid routing (not softmax): scores are independent, selection adds the
        bias, weights come from the unbiased scores and renormalize. The router gemv
        runs in fp32 to match transformers' `F.linear(x.float(), w.float())`."""
        logits = self.gate(x.astype(mx.float32)).astype(mx.float32)
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
        routed = routed * self.scaling
        shared = self.shared_expert(x)
        if self.fp32_combine:
            return (routed.astype(mx.float32) + shared.astype(mx.float32)).astype(x.dtype)
        return routed + shared


class Hy3Attention(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
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

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
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
            queries,
            keys,
            values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else mask,
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))


class Hy3Block(nn.Module):
    def __init__(self, config: Hy3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Hy3Attention(config)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Hy3SparseMoe(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok

    def _fused_step_applies(self) -> bool:
        mlp = self.mlp
        if not isinstance(mlp, Hy3SparseMoe):
            return False
        gate_up = mlp.switch_mlp.gate_up_proj
        down = mlp.switch_mlp.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                mlp.hidden, mlp.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(mlp.split + mlp.k, mlp.k)
            and not mlp.fp32_combine
        )

    def _join(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> tuple[mx.array, mx.array]:
        """(x + attention, its post-norm). At T=1 one kernel does the pair."""
        if ADD_RMS_NORM_KERNEL and x.shape[1] == 1 and add_rms_norm_applies(self.hidden):
            normed = mx.fast.rms_norm(x, self.input_layernorm.weight, self.eps)
            return add_rms_norm(
                x,
                self.self_attn(normed, mask, cache),
                self.post_attention_layernorm.weight,
                self.eps,
            )
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return attended, self.post_attention_layernorm(attended)

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended, h = self._join(x, mask, cache)
        if x.shape[1] == 1 and self._fused_step_applies():
            mlp = self.mlp
            assert isinstance(mlp, Hy3SparseMoe)
            gate_up = mlp.switch_mlp.gate_up_proj
            down = mlp.switch_mlp.down_proj
            assert isinstance(gate_up, QuantizedSwitchLinear)
            assert isinstance(down, QuantizedSwitchLinear)
            assert gate_up.biases is not None and down.biases is not None
            chosen, weights = sigmoid_topk(
                mlp.gate(h.astype(mx.float32)).astype(h.dtype).reshape(-1),
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


class Hy3Trunk(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Hy3Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Hy3Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Hy3(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Hy3Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.config.num_hidden_layers)]

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Hy3Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        mask: mx.array | str | None = None if length == 1 else "causal"
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, mask, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Hy3Activations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


# --- checkpoint block ---

_DENSE = "dense"
_SPARSE = "sparse"


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: NotRequired[float]
    tie_word_embeddings: bool
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    router_scaling_factor: float
    enable_moe_fp32_combine: NotRequired[bool]
    mlp_layer_types: NotRequired[list[str] | None]
    first_k_dense_replace: NotRequired[int]
    eos_token_id: NotRequired[int | list[int] | None]
    rope_parameters: NotRequired[dict[str, object]]


def _config(path: Path) -> Hy3Config:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "hy_v3":
        raise ValueError(f"expected model_type hy_v3, got {raw['model_type']!r}")
    layers = raw["num_hidden_layers"]
    mlp_raw = raw.get("mlp_layer_types")
    if mlp_raw is not None:
        mlp_layer_types = tuple(mlp_raw)
    else:
        first_k = raw.get("first_k_dense_replace", 1)
        mlp_layer_types = (_DENSE,) * first_k + (_SPARSE,) * (layers - first_k)
    if len(mlp_layer_types) != layers:
        raise ValueError(
            f"mlp_layer_types has {len(mlp_layer_types)} entries, expected {layers}"
        )
    eos = raw.get("eos_token_id")
    if isinstance(eos, list):
        eos_tuple = tuple(eos)
    elif eos is None:
        eos_tuple = ()
    else:
        eos_tuple = (eos,)
    rope_params = raw.get("rope_parameters", {})
    rope_theta = raw.get("rope_theta", rope_params.get("rope_theta", 11_158_840.0))
    assert isinstance(rope_theta, (int, float))
    return Hy3Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=layers,
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=rope_theta,
        tie_word_embeddings=raw["tie_word_embeddings"],
        intermediate_size=raw["intermediate_size"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        num_shared_experts=raw["num_shared_experts"],
        router_scaling_factor=raw["router_scaling_factor"],
        enable_moe_fp32_combine=raw.get("enable_moe_fp32_combine", True),
        mlp_layer_types=mlp_layer_types,
        eos_token_id=eos_tuple,
    )


def _drop_mtp(weights: dict[str, mx.array], num_layers: int) -> dict[str, mx.array]:
    """transformers drops `model.layers.80.*` (`_keys_to_ignore_on_load_unexpected`).
    The tree declares 80 layers (0-79); layer 80 is the MTP head, dropped at load."""
    mtp_prefix = f"model.layers.{num_layers}."
    for key in list(weights):
        if key.startswith(mtp_prefix):
            del weights[key]
    return weights


def _convert_experts(
    weights: dict[str, mx.array], config: Hy3Config
) -> dict[str, mx.array]:
    """Experts into the tree's `switch_mlp` leaves, gate‖up row-interleaved.

    The raw HF checkpoint ships 3D stacked tensors: `experts.gate_up_proj`
    `[E, 2*inner, hidden]` (gate‖up concatenated, not interleaved) and
    `experts.down_proj` `[E, hidden, inner]`. A per-expert conversion ships
    `experts.{e}.gate_proj.weight` / `experts.{e}.up_proj.weight` / `.down_proj.weight`
    (the Laguna layout). Both are handled."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != _SPARSE:
            continue
        prefix = f"model.layers.{layer}.mlp."
        inner = config.moe_intermediate_size
        e = config.num_experts

        # e_score_correction_bias: checkpoint keeps it on HYV3MoE (mlp.*), not on
        # HYV3Experts (mlp.experts.*). A per-expert conversion may nest it under
        # experts.* — rename to match the tree.
        bias_key = f"{prefix}experts.e_score_correction_bias"
        if bias_key in weights:
            weights[f"{prefix}e_score_correction_bias"] = weights.pop(bias_key)

        for suffix in ("weight", "scales", "biases"):
            # Case 1: raw HF — single 3D `experts.gate_up_proj` (concatenated, not
            # interleaved). Split, interleave, rename.
            stacked_key = f"{prefix}experts.gate_up_proj.{suffix}"
            if stacked_key in weights:
                gate_up = weights.pop(stacked_key)
                gate, up = gate_up[:, :inner], gate_up[:, inner:]
                interleaved = mx.stack([gate, up], axis=2).reshape(e, 2 * inner, -1)
                mx.eval(interleaved)
                weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = interleaved
                continue

            # Case 2: already-stacked conversion — `switch_mlp.gate_proj` and
            # `switch_mlp.up_proj` as separate `[E, inner, ·]` tensors.
            split_keys = [
                f"{prefix}switch_mlp.{name}_proj.{suffix}" for name in ("gate", "up")
            ]
            if all(key in weights for key in split_keys):
                gates, ups = (weights.pop(key) for key in split_keys)
                interleaved = mx.stack([gates, ups], axis=2).reshape(e, 2 * inner, -1)
                mx.eval(interleaved)
                weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = interleaved
                continue

            # Case 3: per-expert — `experts.{e}.gate_proj` / `experts.{e}.up_proj`.
            gate_keys = [f"{prefix}experts.{i}.gate_proj.{suffix}" for i in range(e)]
            up_keys = [f"{prefix}experts.{i}.up_proj.{suffix}" for i in range(e)]
            if all(key in weights for key in gate_keys + up_keys):
                gates = mx.stack([weights.pop(key) for key in gate_keys])
                ups = mx.stack([weights.pop(key) for key in up_keys])
                interleaved = mx.stack([gates, ups], axis=2).reshape(e, 2 * inner, -1)
                mx.eval(interleaved)
                weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = interleaved

        for suffix in ("weight", "scales", "biases"):
            # down_proj: rename from experts.* to switch_mlp.* (no interleave needed).
            stacked_key = f"{prefix}experts.down_proj.{suffix}"
            if stacked_key in weights:
                weights[f"{prefix}switch_mlp.down_proj.{suffix}"] = weights.pop(
                    stacked_key
                )
                continue

            # Per-expert down_proj.
            down_keys = [f"{prefix}experts.{i}.down_proj.{suffix}" for i in range(e)]
            if all(key in weights for key in down_keys):
                stacked = mx.stack([weights.pop(key) for key in down_keys])
                mx.eval(stacked)
                weights[f"{prefix}switch_mlp.down_proj.{suffix}"] = stacked

    return weights


def _fuse_shared(weights: dict[str, mx.array], config: Hy3Config) -> dict[str, mx.array]:
    """Shared expert's gate‖up concatenated on the output axis, like the dense MLP."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != _SPARSE:
            continue
        prefix = f"model.layers.{layer}.mlp.shared_experts."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _weights(
    directory: Path, config: Hy3Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """`mlp.gate.weight` and `e_score_correction_bias` ship float32 in a bfloat16
    checkpoint and stay float32: the router gemv runs in fp32 to match transformers
    (`F.linear(x.float(), w.float())`), and the bias is fp32 (`_keep_in_fp32_modules_strict`)."""
    weights = load_shards(directory)
    reject_dtype_cast(dtype, weights)
    weights = _drop_mtp(weights, config.num_hidden_layers)

    if dtype is not None:
        weights = {
            key: value
            if key.endswith("e_score_correction_bias")
            or key.endswith("mlp.gate.weight")
            else value.astype(dtype)
            for key, value in weights.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(weights)

    weights = fuse_qkv(weights, config.num_hidden_layers)
    weights = concat_gate_up(weights, config.num_hidden_layers)
    weights = _convert_experts(weights, config)
    return _fuse_shared(weights, config)


def _composite(directory: Path, model: Hy3) -> LanguageModel[ModelInput]:
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
    ), _config, Hy3, _weights, _composite)
