from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import checkpoint, load_shards, prepare_weights, stop_tokens
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.gemma3.tokenizer import Gemma3Tokenizer
from mlx_omnia.engine.models.phi3.config import Phi3Config
from mlx_omnia.engine.models.phi3.model import Phi3


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
