from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.ernie4_5_moe.config import Ernie45MoEConfig
from mlx_omnia.engine.models.ernie4_5_moe.model import Ernie45MoE

DROPPED = (
    "mtp_block.",
    "mtp_linear_proj.",
    "mtp_hidden_norm.",
    "mtp_emb_norm.",
    "e_score_correction_bias",
)


def weights(
    directory: Path, config: Ernie45MoEConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            _drop_mtp,
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: stack_experts(weights, layers, config.moe_num_experts),
            lambda weights: interleave_gate_up(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
        ],
        dtype,
    )


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {
        key: value
        for key, value in weights.items()
        if not any(pattern in key for pattern in DROPPED)
    }


def _composite(directory: Path, model: Ernie45MoE) -> LanguageModel[ModelInput]:
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
    Ernie45MoEConfig,
    Ernie45MoE,
    weights,
    _composite,
    model_types=("ernie4_5_moe",),
)
