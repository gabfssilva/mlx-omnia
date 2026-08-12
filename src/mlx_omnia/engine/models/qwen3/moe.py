"""The sparse checkpoint: `qwen3_moe`.

Property names are the checkpoint's (mlx export layout: experts stacked as
`mlp.switch_mlp.*`). qkv is fused on the output axis and gate‖up row-interleaved
at load, dict-side.
"""

from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.qwen3.config import Qwen3MoEConfig
from mlx_omnia.engine.models.qwen3.model import Qwen3MoE

__all__ = ["CHECKPOINT", "Qwen3MoE", "Qwen3MoEConfig", "weights"]


def weights(
    directory: Path, config: Qwen3MoEConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: interleave_gate_up(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: Qwen3MoE) -> LanguageModel[ModelInput]:
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
    Qwen3MoEConfig,
    Qwen3MoE,
    weights,
    _composite,
    model_types=("qwen3_moe",),
)
