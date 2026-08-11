from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.lfm2.config import LFM2MoEConfig
from sideros.models.lfm2.moe.model import LFM2MoE


def weights(
    directory: Path, config: LFM2MoEConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """`expert_bias` ships float32 in a bfloat16 checkpoint and stays float32: the router
    adds it in float32 whatever the model's precision."""
    loaded = load_shards(directory)
    reject_dtype_cast(dtype, loaded)

    if dtype is not None:
        loaded = {
            key: value if key.endswith("expert_bias") else value.astype(dtype)
            for key, value in loaded.items()
        }
    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    loaded = _fuse_dense_mlp(loaded, config.num_dense_layers)
    return _stack_experts(loaded, config)


def _fuse_dense_mlp(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    for layer in range(layers):
        prefix = f"model.layers.{layer}.feed_forward."
        keys = [f"{prefix}{name}.weight" for name in ("w1", "w3")]
        if not all(key in weights for key in keys):
            continue
        fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
        mx.eval(fused)
        weights[f"{prefix}w13.weight"] = fused
    return weights


def _stack_experts(weights: dict[str, mx.array], config: LFM2MoEConfig) -> dict[str, mx.array]:
    """One `[experts, out, in]` tensor per projection, w1‖w3 joined on the output axis.
    The per-expert slices leave the dict, or both copies stay resident."""
    for layer in range(config.num_dense_layers, config.num_hidden_layers):
        prefix = f"model.layers.{layer}.feed_forward.experts."
        parts: dict[str, list[mx.array]] = {}
        for name in ("w1", "w3", "w2"):
            parts[name] = [
                weights.pop(f"{prefix}{expert}.{name}.weight")
                for expert in range(config.num_experts)
            ]
        w13 = mx.stack(
            [mx.concatenate([g, u], axis=0) for g, u in zip(parts["w1"], parts["w3"], strict=True)]
        )
        w2 = mx.stack(parts["w2"])
        mx.eval(w13, w2)
        weights[f"{prefix}w13.weight"] = w13
        weights[f"{prefix}w2.weight"] = w2
    return weights


def _composite(directory: Path, model: LFM2MoE) -> LanguageModel[ModelInput]:
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
    LFM2MoEConfig,
    LFM2MoE,
    weights,
    _composite,
    model_types=("lfm2_moe",),
)
