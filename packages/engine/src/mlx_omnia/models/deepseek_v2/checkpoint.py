from pathlib import Path

import mlx.core as mx

from mlx_omnia.bpe import ByteLevelBPE
from mlx_omnia.chat import chat_capabilities
from mlx_omnia.checkpoint import (
    checkpoint,
    concat_gate_up,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.language import LanguageModel, TextLanguageModel
from mlx_omnia.model import CompositeModel, ModelInput
from mlx_omnia.models.deepseek_v2.config import DeepseekV2Config
from mlx_omnia.models.deepseek_v2.model import DeepseekV2


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
