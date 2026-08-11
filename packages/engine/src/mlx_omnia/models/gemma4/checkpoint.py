from pathlib import Path

import mlx.core as mx

from mlx_omnia.chat import chat_capabilities
from mlx_omnia.checkpoint import (
    checkpoint,
    drop_tied_head,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.core.masks import FULL
from mlx_omnia.language import LanguageModel, TextLanguageModel
from mlx_omnia.model import CompositeModel, ModelInput
from mlx_omnia.models.gemma4.config import Gemma4Config
from mlx_omnia.models.gemma4.model import Gemma4


def weights(
    directory: Path, config: Gemma4Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    loaded: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        reject_dtype_cast(dtype, part)
        for name, array in part.items():
            renamed = _renamed(name)
            if renamed is None:
                continue
            loaded[renamed] = array.astype(dtype) if dtype is not None else array

    if config.tied:
        drop_tied_head(loaded)

    _drop_unused_attn_weights(loaded, config)

    # No qkv fusion, no gate_up concat, no _fold_norm_scales.
    return loaded


def _renamed(name: str) -> str | None:
    """Extract the text tower from a multimodal checkpoint.

    Raw HF `Gemma4ForConditionalGeneration`: `model.language_model.*` for the text
    tower, `model.vision_tower.*` / `model.audio_*` for the modality towers. We keep
    only `model.language_model.*` → `model.*` and discard everything else.
    Standalone text checkpoints already use `model.*` and pass through unchanged.
    """
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    # Standalone text checkpoints: model.* (but NOT model.vision_tower.*, etc.).
    if name.startswith("model."):
        suffix = name.removeprefix("model.")
        if suffix.startswith((
            "vision_tower", "multi_modal_projector", "audio_tower",
            "embed_audio", "embed_vision", "vision_embedder", "visual",
        )):
            return None
        return name
    # lm_head at top level (untied) stays.
    if name.startswith("lm_head."):
        return name
    # Everything else (vision_tower, audio, multimodal_projector, etc.) is discarded.
    return None


def _drop_unused_attn_weights(
    weights: dict[str, mx.array], config: Gemma4Config
) -> None:
    """Drop attention weights the tree does not declare, so strict load stays exact.

    - KV-shared layers (idx >= first_kv_shared): no k_proj/v_proj/k_norm/v_norm.
    - k_eq_v full layers: no v_proj (v = k).
    """
    text = config.text_config
    types = text.attention_types
    for layer_idx in range(text.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.self_attn."
        is_shared = text.is_kv_shared_layer(layer_idx)
        is_k_eq_v_full = text.attention_k_eq_v and types[layer_idx] == FULL
        drop = set[str]()
        if is_shared:
            drop.update(("k_proj", "v_proj", "k_norm", "v_norm"))
        elif is_k_eq_v_full:
            drop.add("v_proj")
        for name in drop:
            for suffix in ("weight", "scales", "biases"):
                weights.pop(f"{prefix}{name}.{suffix}", None)


def _composite(directory: Path, model: Gemma4) -> LanguageModel[ModelInput]:
    from mlx_omnia.models.gemma3.tokenizer import Gemma3Tokenizer

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
    Gemma4Config,
    Gemma4,
    weights,
    _composite,
    model_types=("gemma4", "gemma4_unified", "gemma4_assistant"),
)
