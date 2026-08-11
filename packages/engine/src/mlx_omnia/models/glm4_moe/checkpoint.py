from pathlib import Path

import mlx.core as mx

from mlx_omnia.bpe import ByteLevelBPE
from mlx_omnia.chat import chat_capabilities
from mlx_omnia.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.language import LanguageModel, TextLanguageModel
from mlx_omnia.model import CompositeModel, ModelInput
from mlx_omnia.models.glm4_moe.config import Glm4MoEConfig
from mlx_omnia.models.glm4_moe.model import Glm4MoE


def drop_extra_layers(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """The MTP block ships as one layer past the trunk's last index."""
    return {
        key: value
        for key, value in weights.items()
        if not key.startswith(f"model.layers.{layers}.")
    }


def weights(directory: Path, config: Glm4MoEConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: drop_extra_layers(weights, layers),
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: stack_experts(weights, layers, config.n_routed_experts),
            lambda weights: interleave_gate_up(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: Glm4MoE) -> LanguageModel[ModelInput]:
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
    Glm4MoEConfig,
    Glm4MoE,
    weights,
    _composite,
    model_types=("glm4_moe",),
)
