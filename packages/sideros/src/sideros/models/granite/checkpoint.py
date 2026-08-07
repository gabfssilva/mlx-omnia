from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.granite.config import GraniteConfig
from sideros.models.granite.model import Granite


def weights(directory: Path, config: GraniteConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: Granite) -> LanguageModel[ModelInput]:
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
    GraniteConfig,
    Granite,
    weights,
    _composite,
    model_types=("granite",),
)
