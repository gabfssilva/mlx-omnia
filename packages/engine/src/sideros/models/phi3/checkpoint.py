from pathlib import Path

import mlx.core as mx

from sideros.chat import chat_capabilities
from sideros.checkpoint import checkpoint, load_shards, prepare_weights, stop_tokens
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.gemma3.tokenizer import Gemma3Tokenizer
from sideros.models.phi3.config import Phi3Config
from sideros.models.phi3.model import Phi3


def weights(directory: Path, config: Phi3Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    return prepare_weights(config, load_shards(directory), [], dtype)


def _composite(directory: Path, model: Phi3) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            Gemma3Tokenizer.from_file(directory / "tokenizer.json", bos="<s>"),
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
    Phi3Config,
    Phi3,
    weights,
    _composite,
    model_types=("phi3",),
)
