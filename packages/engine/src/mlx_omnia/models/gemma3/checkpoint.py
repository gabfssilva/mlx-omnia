from pathlib import Path

import mlx.core as mx

from mlx_omnia.chat import chat_capabilities
from mlx_omnia.checkpoint import (
    checkpoint,
    concat_gate_up,
    fold_norm_scales,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from mlx_omnia.language import LanguageModel, TextLanguageModel
from mlx_omnia.model import CompositeModel, ModelInput
from mlx_omnia.models.gemma3.config import Gemma3TextConfig
from mlx_omnia.models.gemma3.model import Gemma3
from mlx_omnia.models.gemma3.tokenizer import Gemma3Tokenizer


def weights(
    directory: Path, config: Gemma3TextConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
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


def _composite(directory: Path, model: Gemma3) -> LanguageModel[ModelInput]:
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
    Gemma3TextConfig,
    Gemma3,
    weights,
    _composite,
    model_types=("gemma3_text",),
)
