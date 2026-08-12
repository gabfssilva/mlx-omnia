from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import checkpoint, drop_tied_head, reject_dtype_cast, stop_tokens
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.model import Mamba2


def weights(
    directory: Path,
    config: Mamba2Config,
    dtype: mx.Dtype | None,
) -> dict[str, mx.array]:
    """The checkpoint's tensors in the tree's names and layout, up to (not
    including) the tree itself."""
    loaded: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        reject_dtype_cast(dtype, part)
        for name, array in part.items():
            renamed = _renamed(name)
            if renamed is None:
                continue
            # A_log, dt_bias, D stay float32 at every precision: the decay
            # saturates if they round-trip through bf16.
            keep_fp32 = (
                renamed.endswith("A_log")
                or renamed.endswith("dt_bias")
                or renamed.endswith(".D")
            )
            if dtype is not None and not keep_fp32:
                loaded[renamed] = array.astype(dtype)
            else:
                loaded[renamed] = array

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    # The torch conv layout `[conv_dim, 1, kernel]` marks a raw HF checkpoint;
    # an mlx conversion arrives as `[conv_dim, kernel, 1]`. Both squeeze to
    # `[conv_dim, kernel]`.
    for name, array in loaded.items():
        if name.endswith("conv1d.weight"):
            if array.ndim == 3 and array.shape[1] == 1:
                loaded[name] = array.squeeze(1)
            elif array.ndim == 3 and array.shape[2] == 1:
                loaded[name] = array.squeeze(2)

    return loaded


def _renamed(name: str) -> str | None:
    """Two dialects: raw HF `backbone.*` vs mlx `model.*`. The HF names
    `embeddings`/`norm_f` map to the tree's `embed_tokens`/`norm`."""
    if name.startswith("backbone."):
        rest = name.removeprefix("backbone.")
        rest = rest.replace("embeddings.", "embed_tokens.", 1)
        rest = rest.replace("norm_f.", "norm.", 1)
        return "model." + rest
    if name.startswith("model."):
        rest = name.removeprefix("model.")
        rest = rest.replace("embeddings.", "embed_tokens.", 1)
        rest = rest.replace("norm_f.", "norm.", 1)
        return "model." + rest
    return name


def _composite(directory: Path, model: Mamba2) -> LanguageModel[ModelInput]:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    return CompositeModel(
        TextLanguageModel(
            model,
            tokenizer,
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
    Mamba2Config,
    Mamba2,
    weights,
    _composite,
    model_types=("mamba2",),
)
