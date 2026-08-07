from pathlib import Path

import mlx.core as mx

from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    fold_norm_scales,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.gemma.config import GemmaConfig
from sideros.models.gemma.model import Gemma
from sideros.models.gemma3.tokenizer import Gemma3Tokenizer


def weights(directory: Path, config: GemmaConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
            fold_norm_scales,
        ],
        dtype,
    )


def _composite(directory: Path, model: Gemma) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            Gemma3Tokenizer.from_file(directory / "tokenizer.json"),
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
    GemmaConfig,
    Gemma,
    weights,
    _composite,
    model_types=("gemma",),
)
