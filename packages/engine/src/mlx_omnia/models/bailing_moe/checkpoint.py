from collections.abc import Callable
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
from mlx_omnia.models.bailing_moe.config import BailingMoEConfig
from mlx_omnia.models.bailing_moe.model import BailingMoE


def weights(
    directory: Path, config: BailingMoEConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    passes: list[Callable[[dict[str, mx.array]], dict[str, mx.array]]] = [
        lambda weights: stack_experts(weights, layers, config.num_experts),
        lambda weights: interleave_gate_up(weights, layers),
        lambda weights: concat_gate_up(weights, layers),
    ]
    if config.norm_head and not config.tie_word_embeddings:
        passes.append(_normalize_head)
    return prepare_weights(config, load_shards(directory), passes, dtype)


def _normalize_head(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    head = weights["lm_head.weight"]
    norm = mx.linalg.norm(head.astype(mx.float32), axis=0, keepdims=True) + 1e-7
    weights["lm_head.weight"] = (head / norm).astype(head.dtype)
    mx.eval(weights["lm_head.weight"])
    return weights


def _composite(directory: Path, model: BailingMoE) -> LanguageModel[ModelInput]:
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
    BailingMoEConfig,
    BailingMoE,
    weights,
    _composite,
    model_types=("bailing_moe",),
)
