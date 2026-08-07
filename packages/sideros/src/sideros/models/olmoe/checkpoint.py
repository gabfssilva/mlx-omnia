from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.olmoe.config import OlmoEConfig
from sideros.models.olmoe.model import OlmoE


def weights(directory: Path, config: OlmoEConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: stack_experts(weights, layers, config.num_experts),
            lambda weights: interleave_gate_up(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: OlmoE) -> LanguageModel[ModelInput]:
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
    OlmoEConfig,
    OlmoE,
    weights,
    _composite,
    model_types=("olmoe",),
)
