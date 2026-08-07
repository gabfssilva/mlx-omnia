from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.deepseek_v2.config import DeepseekV2Config
from sideros.models.deepseek_v2.model import DeepseekV2


def weights(
    directory: Path, config: DeepseekV2Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: stack_experts(weights, layers, config.n_routed_experts),
            lambda weights: interleave_gate_up(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: DeepseekV2) -> LanguageModel[ModelInput]:
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
    DeepseekV2Config,
    DeepseekV2,
    weights,
    _composite,
    model_types=("deepseek_v2",),
)
