from pathlib import Path

import mlx.core as mx

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
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.hy3.config import SPARSE, Hy3Config
from sideros.models.hy3.model import Hy3


def weights(
    directory: Path, config: Hy3Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """`mlp.gate.weight` and `e_score_correction_bias` ship float32 in a bfloat16
    checkpoint and stay float32: the router gemv runs in fp32 to match transformers
    (`F.linear(x.float(), w.float())`), and the bias is fp32 (`_keep_in_fp32_modules_strict`)."""
    loaded = load_shards(directory)
    reject_dtype_cast(dtype, loaded)
    loaded = _drop_mtp(loaded, config.num_hidden_layers)

    if dtype is not None:
        loaded = {
            key: value
            if key.endswith("e_score_correction_bias") or key.endswith("mlp.gate.weight")
            else value.astype(dtype)
            for key, value in loaded.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    loaded = concat_gate_up(loaded, config.num_hidden_layers)
    loaded = _convert_experts(loaded, config)
    return _fuse_shared(loaded, config)


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
    for layer, kind in enumerate(config.layer_types):
        if kind != SPARSE:
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
    for layer, kind in enumerate(config.layer_types):
        if kind != SPARSE:
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


def _composite(directory: Path, model: Hy3) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos),
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
    Hy3Config,
    Hy3,
    weights,
    _composite,
    model_types=("hy_v3",),
)
