from pathlib import Path

import mlx.core as mx

from mlx_omnia.bpe import ByteLevelBPE
from mlx_omnia.chat import chat_capabilities
from mlx_omnia.checkpoint import (
    checkpoint,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    stop_tokens,
)
from mlx_omnia.language import LanguageModel, TextLanguageModel
from mlx_omnia.model import CompositeModel, ModelInput
from mlx_omnia.models.lfm2.config import LFM2Config
from mlx_omnia.models.lfm2.dense.model import LFM2


def weights(directory: Path, config: LFM2Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    loaded = load_shards(directory)
    if dtype is not None:
        loaded = {key: value.astype(dtype) for key, value in loaded.items()}
    if config.tie_word_embeddings:
        drop_tied_head(loaded)
    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    return _fuse_dense_mlp(loaded, config.num_hidden_layers)


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


def _composite(directory: Path, model: LFM2) -> LanguageModel[ModelInput]:
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
    LFM2Config,
    LFM2,
    weights,
    _composite,
    model_types=("lfm2",),
)
