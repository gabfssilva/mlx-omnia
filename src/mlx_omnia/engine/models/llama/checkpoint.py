from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.llama.config import LlamaConfig
from mlx_omnia.engine.models.llama.model import Llama


def weights(directory: Path, config: LlamaConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            _drop_inv_freq,
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
        ],
        dtype,
    )


def _drop_inv_freq(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Llama-2 ships one precomputed rope table per layer. They are recomputed from
    `rope_theta`, so they are dead weights the tree never declares."""
    return {key: value for key, value in weights.items() if "rotary_emb.inv_freq" not in key}


def _composite(directory: Path, model: Llama) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
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
    LlamaConfig,
    Llama,
    weights,
    _composite,
    model_types=("llama",),
)
