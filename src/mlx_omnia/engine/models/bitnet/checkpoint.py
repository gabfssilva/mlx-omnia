from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    drop_tied_head,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.bitnet.config import BitNetConfig
from mlx_omnia.engine.models.bitnet.model import BitNet


def weights(
    directory: Path, config: BitNetConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """The packed ternary ``.weight`` is uint8 and must stay uint8: a ``dtype=`` cast
    keeps the shape (so ``load_weights`` accepts it) and destroys the codes. Cast
    everything else — ``weight_scale``, the embed table, the norms — and leave the
    packed weights alone. No fusions: each projection carries its own per-tensor
    ``weight_scale``, so qkv or gate‖up fusion would collapse independent scales."""
    loaded = load_shards(directory)
    reject_dtype_cast(dtype, loaded)
    if dtype is not None:
        loaded = {
            key: value if value.dtype == mx.uint8 else value.astype(dtype)
            for key, value in loaded.items()
        }
    if config.tie_word_embeddings:
        drop_tied_head(loaded)
    return loaded


def _composite(directory: Path, model: BitNet) -> LanguageModel[ModelInput]:
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
    BitNetConfig,
    BitNet,
    weights,
    _composite,
    model_types=("bitnet",),
)
