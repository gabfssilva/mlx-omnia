from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    concat_gate_up,
    fold_norm_scales,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.gemma2.config import Gemma2Config
from mlx_omnia.engine.models.gemma2.model import Gemma2
from mlx_omnia.engine.models.gemma3.tokenizer import Gemma3Tokenizer


def weights(directory: Path, config: Gemma2Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
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


def _composite(directory: Path, model: Gemma2) -> LanguageModel[ModelInput]:
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
    Gemma2Config,
    Gemma2,
    weights,
    _composite,
    model_types=("gemma2",),
)
