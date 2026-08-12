from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.jamba.config import JambaConfig
from mlx_omnia.engine.models.jamba.model import Jamba


def _squeeze_conv(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    for key, value in weights.items():
        if key.endswith("conv1d.weight") and value.ndim == 3:
            weights[key] = value.reshape(value.shape[0], -1)
    return weights


def _stack_feed_forward(weights: dict[str, mx.array], config: JambaConfig) -> dict[str, mx.array]:
    """Jamba's experts hang off `feed_forward`, not `mlp`, so the shared pass is aimed
    there and the interleave is done by hand on the same prefix."""
    layers = config.num_hidden_layers
    weights = stack_experts(weights, layers, config.num_experts, prefix="feed_forward")
    for layer in range(layers):
        prefix = f"model.layers.{layer}.feed_forward.switch_mlp."
        keys = [f"{prefix}{name}_proj.weight" for name in ("gate", "up")]
        if not all(key in weights for key in keys):
            continue
        parts = [weights.pop(key) for key in keys]
        experts, rows, cols = parts[0].shape
        fused = mx.stack(parts, axis=2).reshape(experts, 2 * rows, cols)
        mx.eval(fused)
        weights[f"{prefix}gate_up_proj.weight"] = fused
    return weights


def _fuse_dense(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    for layer in range(layers):
        prefix = f"model.layers.{layer}.feed_forward."
        keys = [f"{prefix}{name}_proj.weight" for name in ("gate", "up")]
        if not all(key in weights for key in keys):
            continue
        fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
        mx.eval(fused)
        weights[f"{prefix}gate_up_proj.weight"] = fused
    return weights


def _weights(directory: Path, config: JambaConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            _squeeze_conv,
            lambda weights: _stack_feed_forward(weights, config),
            lambda weights: _fuse_dense(weights, layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: Jamba) -> LanguageModel[ModelInput]:
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
    JambaConfig,
    Jamba,
    _weights,
    _composite,
    model_types=("jamba",),
)
