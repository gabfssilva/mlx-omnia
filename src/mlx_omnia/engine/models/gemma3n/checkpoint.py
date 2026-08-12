from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import checkpoint, load_shards, prepare_weights, stop_tokens
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.gemma3.tokenizer import Gemma3Tokenizer
from mlx_omnia.engine.models.gemma3n.config import Gemma3nConfig
from mlx_omnia.engine.models.gemma3n.model import Gemma3n

TOWERS = ("model.vision_tower.", "model.audio_tower.", "model.embed_vision.",
          "model.embed_audio.", "vision_tower.", "audio_tower.")
NESTED = "model.language_model."


def weights(directory: Path, config: Gemma3nConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    return prepare_weights(
        config,
        load_shards(directory),
        [_flatten, lambda tensors: _clip_altup(tensors, config)],
        dtype,
    )


def _flatten(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Drop the towers and lift the text trunk back to `model.*`."""
    lifted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith(TOWERS):
            continue
        lifted[key.replace(NESTED, "model.", 1) if key.startswith(NESTED) else key] = value
    return lifted


def _clip_altup(weights: dict[str, mx.array], config: Gemma3nConfig) -> dict[str, mx.array]:
    """The reference clips the two coefficient matrices on every call; the clip is a
    function of the weights alone, so it happens once here."""
    clip = config.text_config.altup_coef_clip
    if clip is None:
        return weights
    for key, value in weights.items():
        if key.endswith(("altup.prediction_coefs.weight", "altup.correction_coefs.weight")):
            weights[key] = mx.clip(value, -clip, clip)
    mx.eval(list(weights.values()))
    return weights


def _composite(directory: Path, model: Gemma3n) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            Gemma3Tokenizer.from_file(directory / "tokenizer.json"),
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
    Gemma3nConfig,
    Gemma3n,
    weights,
    _composite,
    model_types=("gemma3n",),
)
