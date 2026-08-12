from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import checkpoint, reject_dtype_cast
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.gpt2.config import GPT2Config
from mlx_omnia.engine.models.gpt2.model import GPT2
from mlx_omnia.engine.models.gpt2.tokenizer import GPT2Tokenizer


def weights(directory: Path, config: GPT2Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    raw = mx.load(str(directory / "model.safetensors"))
    assert isinstance(raw, dict)
    reject_dtype_cast(dtype, raw)
    # h.{i}.attn.bias is the checkpoint's causal-mask buffer, not a weight.
    return _transpose_conv1d(
        {
            key: value.astype(dtype if dtype is not None else mx.float32)
            for key, value in raw.items()
            if not key.endswith(".attn.bias")
        }
    )


_CONV1D = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")


def _transpose_conv1d(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """GPT-2's Conv1D ships `[in, out]`. This is the one load-time layout change that
    moves the last axis, so it has to land before any grouping: quantizing the raw
    matrix groups along the output columns."""
    return {
        key: value.T if key.endswith(_CONV1D) else value for key, value in weights.items()
    }


def _end_of_text(tokenizer: GPT2Tokenizer) -> tuple[int, ...]:
    end = tokenizer.encoder.get("<|endoftext|>")
    return () if end is None else (end,)


def _composite(directory: Path, model: GPT2) -> LanguageModel[ModelInput]:
    tokenizer = GPT2Tokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
    )
    return CompositeModel(
        TextLanguageModel(model, tokenizer, stop=_end_of_text(tokenizer)),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    GPT2Config,
    GPT2,
    weights,
    _composite,
    model_types=("gpt2",),
)
