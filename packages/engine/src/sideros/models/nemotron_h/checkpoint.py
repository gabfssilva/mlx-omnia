from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import checkpoint, load_shards, prepare_weights, stop_tokens
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.nemotron_h.config import MOE, NemotronHConfig
from sideros.models.nemotron_h.model import NemotronH


def weights(
    directory: Path, config: NemotronHConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    return prepare_weights(
        config,
        load_shards(directory),
        [_drop_mtp, _squeeze_conv, lambda loaded: _stack_experts(loaded, config)],
        dtype,
    )


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {key: value for key, value in weights.items() if not key.startswith("mtp.")}


def _squeeze_conv(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """`[conv_dim, 1, kernel]` (or its transpose) down to `[conv_dim, kernel]`."""
    for key, value in weights.items():
        if not key.endswith("conv1d.weight"):
            continue
        weights[key] = value.reshape(value.shape[0], -1) if value.ndim == 3 else value
    return weights


def _stack_experts(weights: dict[str, mx.array], config: NemotronHConfig) -> dict[str, mx.array]:
    for layer, kind in enumerate(config.pattern):
        if kind != MOE:
            continue
        source = f"backbone.layers.{layer}.mixer.experts."
        target = f"backbone.layers.{layer}.mixer.switch_mlp."
        for name, leaf in (("up_proj", "fc1"), ("down_proj", "fc2")):
            if f"{source}0.{name}.weight" not in weights:
                continue
            stacked = mx.stack(
                [
                    weights.pop(f"{source}{expert}.{name}.weight")
                    for expert in range(config.routed_experts)
                ]
            )
            mx.eval(stacked)
            weights[f"{target}{leaf}.weight"] = stacked
    return weights


def _composite(directory: Path, model: NemotronH) -> LanguageModel[ModelInput]:
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
    NemotronHConfig,
    NemotronH,
    weights,
    _composite,
    model_types=("nemotron_h",),
)
