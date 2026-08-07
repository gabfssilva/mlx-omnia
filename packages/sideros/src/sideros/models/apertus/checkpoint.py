from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.apertus.config import ApertusConfig
from sideros.models.apertus.model import Apertus


def _squeeze_alphas(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """xIELU's two scalars ship with singleton axes; the tree declares them rank 0, and
    `update(strict=True)` compares shapes."""
    for key, value in weights.items():
        if key.endswith(("alpha_p", "alpha_n")):
            weights[key] = value.reshape(())
    return weights


def _weights(directory: Path, config: ApertusConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [lambda weights: fuse_qkv(weights, layers), _squeeze_alphas],
        dtype,
    )


def _composite(directory: Path, model: Apertus) -> LanguageModel[ModelInput]:
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
    ApertusConfig,
    Apertus,
    _weights,
    _composite,
    model_types=("apertus",),
)
