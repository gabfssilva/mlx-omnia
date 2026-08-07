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
from sideros.models.laguna.config import LagunaConfig
from sideros.models.laguna.model import Laguna


def weights(
    directory: Path, config: LagunaConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """e_score_correction_bias ships float32 in a bfloat16 checkpoint and stays
    float32: the router adds it in float32 whatever the model's precision."""
    loaded = _rename_vlm(load_shards(directory))
    reject_dtype_cast(dtype, loaded)

    if dtype is not None:
        loaded = {
            key: value
            if key.endswith("e_score_correction_bias")
            else value.astype(dtype)
            for key, value in loaded.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    loaded = concat_gate_up(loaded, config.num_hidden_layers)
    loaded = _stack_experts(loaded, config)
    loaded = _interleave_stacked_experts(loaded, config)
    return _fuse_shared(loaded, config)


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


def _composite(directory: Path, model: Laguna) -> LanguageModel[ModelInput]:
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
    LagunaConfig,
    Laguna,
    weights,
    _composite,
    model_types=("laguna",),
)
