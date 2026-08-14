from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    interleave_gate_up,
    load_shards,
    materialize,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.qwen3_next.config import Qwen3NextConfig
from mlx_omnia.engine.models.qwen3_next.model import Qwen3Next

ZERO_CENTRED = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def weights(
    directory: Path, config: Qwen3NextConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            _drop_mtp,
            _squeeze_conv,
            lambda weights: stack_experts(weights, layers, config.num_experts),
            lambda weights: interleave_gate_up(weights, layers),
            _fold_norm_scales,
        ],
        dtype,
    )


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {key: value for key, value in weights.items() if "mtp." not in key}


def _squeeze_conv(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    for key, value in weights.items():
        if key.endswith("conv1d.weight") and value.ndim == 3:
            weights[key] = value.reshape(value.shape[0], -1)
    return weights


def _fold_norm_scales(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    for key, value in weights.items():
        if value.ndim == 1 and (key.endswith(ZERO_CENTRED) or key == "model.norm.weight"):
            weights[key] = value + 1
    materialize(list(weights.values()))
    return weights


def _composite(directory: Path, model: Qwen3Next) -> LanguageModel[ModelInput]:
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
    Qwen3NextConfig,
    Qwen3Next,
    weights,
    _composite,
    model_types=("qwen3_next",),
)
