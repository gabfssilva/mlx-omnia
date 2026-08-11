"""The dense checkpoint: `qwen3`."""

from pathlib import Path

import mlx.core as mx

from mlx_omnia.bpe import ByteLevelBPE
from mlx_omnia.chat import chat_capabilities
from mlx_omnia.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from mlx_omnia.language import LanguageModel, TextLanguageModel
from mlx_omnia.model import CompositeModel, ModelInput
from mlx_omnia.models.qwen3.config import Qwen3Config
from mlx_omnia.models.qwen3.model import Qwen3


def weights(directory: Path, config: Qwen3Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
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


def _composite(directory: Path, model: Qwen3) -> LanguageModel[ModelInput]:
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
    Qwen3Config,
    Qwen3,
    weights,
    _composite,
    model_types=("qwen3",),
)
