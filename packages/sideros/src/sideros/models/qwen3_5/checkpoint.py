from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import (
    ChatCapability,
    MultimodalChatCapability,
    chat_template,
)
from sideros.checkpoint import (
    checkpoint,
    drop_tied_head,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.language import LanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.qwen3_5.config import Qwen35Config, Qwen35TextConfig
from sideros.models.qwen3_5.model import Qwen35, Qwen35LanguageModel
from sideros.models.qwen3_5.vision import (
    load_processor_config,
    normalized_patch_weight,
)


def weights(
    directory: Path,
    config: Qwen35Config,
    dtype: mx.Dtype | None,
) -> dict[str, mx.array]:
    """The checkpoint's tensors in the tree's names and layout, up to (not including) the
    tree itself: the quantizing load needs this dict on its own."""
    loaded: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        reject_dtype_cast(dtype, part)
        for name, array in part.items():
            renamed = _renamed(name)
            if renamed is None:
                continue
            # A_log stays float32 at every precision: the decay is computed there.
            cast = dtype is not None and not renamed.endswith("A_log")
            loaded[renamed] = array.astype(dtype) if dtype is not None and cast else array

    if config.tied:
        drop_tied_head(loaded)

    # The torch conv layout `[dim, 1, kernel]` marks a raw HF checkpoint: its RMSNorms
    # are still zero-centered (scale = 1 + w, as in Gemma), so the shift bakes in here —
    # after the cast, exactly like transformers' float32 `1.0 + weight`. An mlx
    # conversion arrives as `[dim, kernel, 1]` with the shift already folded in.
    text = config.text_config
    first_linear = text.layer_types.index("linear_attention")
    conv = f"model.layers.{first_linear}.linear_attn.conv1d.weight"
    raw_hf = loaded[conv].shape[1] == 1
    for name, array in loaded.items():
        if name.endswith("conv1d.weight"):
            loaded[name] = array.squeeze(1 if raw_hf else 2)
        elif raw_hf and (name == "model.norm.weight" or name.endswith(_ZERO_CENTERED)):
            loaded[name] = array + 1

    # The tower's Conv3d folds into a matmul, and both weight dialects flatten here.
    patch = "visual.patch_embed.proj.weight"
    if config.vision_config is not None and patch in loaded:
        loaded[patch] = normalized_patch_weight(loaded[patch], config.vision_config)

    loaded = _fuse_projections(loaded, text)
    return _fuse_moe(loaded, text)


_ZERO_CENTERED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def _renamed(name: str) -> str | None:
    """Two dialects land here. Raw HF: `model.language_model.*`, tower under
    `model.visual.*`, MTP head serialized. mlx conversions: `language_model.model.*`,
    tower under `vision_tower.*`, MTP dropped. Both normalize to the house names — the
    tower to `visual.*`; the MTP head is not part of the port and leaves."""
    if name.startswith("mtp."):
        return None
    if name.startswith("vision_tower."):
        return "visual." + name.removeprefix("vision_tower.")
    if name.startswith("model.visual."):
        return "visual." + name.removeprefix("model.visual.")
    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    elif name.startswith("language_model.model."):
        name = "model." + name.removeprefix("language_model.model.")
    elif name.startswith("language_model."):
        name = name.removeprefix("language_model.")
    return name


def _fuse_projections(
    weights: dict[str, mx.array], config: Qwen35TextConfig
) -> dict[str, mx.array]:
    """One fused projection per mixer instead of three or four, concatenated on the
    output axis — row-aligned in every representation, so dense and packed fuse alike.
    The originals leave the dict, otherwise both copies stay resident."""
    for layer, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            prefix, parts = f"model.layers.{layer}.self_attn.", ("q_proj", "k_proj", "v_proj")
        else:
            prefix = f"model.layers.{layer}.linear_attn."
            parts = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{part}.{suffix}" for part in parts]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}fused_proj.{suffix}"] = fused

    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{part}_proj.{suffix}" for part in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _fuse_moe(weights: dict[str, mx.array], config: Qwen35TextConfig) -> dict[str, mx.array]:
    """Two load-time fusions on the sparse block, both row-aligned so packed weights,
    scales and biases take the same path:

    - the shared expert's logit becomes row 256 of the router's matrix, so one gemv
      produces the routing logits and the shared gate together;
    - gate and up interleave row by row ([g0,u0,g1,…]) into the expert stack, and the
      shared expert's pair is stacked as slot 256. That stack was already a copy the
      load paid for; the *down* stack is the one that goes mmap'd from the file into
      the model, and appending a row to it materializes 5.4 GB.
    """
    if config.num_experts == 0:
        return weights
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix in ("weight", "scales", "biases"):
            router = f"{prefix}gate.{suffix}"
            shared_gate = f"{prefix}shared_expert_gate.{suffix}"
            if router in weights and shared_gate in weights:
                fused = mx.concatenate([weights.pop(router), weights.pop(shared_gate)], axis=0)
                mx.eval(fused)
                weights[router] = fused

            keys = [f"{prefix}switch_mlp.{part}_proj.{suffix}" for part in ("gate", "up")]
            shared = [f"{prefix}shared_expert.{part}_proj.{suffix}" for part in ("gate", "up")]
            if not all(key in weights for key in keys + shared):
                continue
            stacked = [weights.pop(key) for key in keys]
            experts, rows, cols = stacked[0].shape
            routed = mx.stack(stacked, axis=2).reshape(experts, 2 * rows, cols)
            pair = [weights.pop(key) for key in shared]
            slot = mx.stack(pair, axis=1).reshape(1, 2 * rows, cols)
            fused = mx.concatenate([routed, slot], axis=0)
            mx.eval(fused)
            weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = fused
    return weights


def _chat(
    directory: Path, facade: Qwen35LanguageModel
) -> list[ChatCapability | MultimodalChatCapability]:
    """Which of the two the checkpoint gets is decided by the vision tower: with one, the
    template's image marker is what the conversation is cut on."""
    template = chat_template(directory)
    if template is None:
        return []
    marker = facade.image_marker
    if marker is None:
        return [ChatCapability(template)]
    return [MultimodalChatCapability(template, marker)]


def _composite(directory: Path, model: Qwen35) -> LanguageModel[ModelInput]:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    processor_path = directory / "preprocessor_config.json"
    processor = load_processor_config(processor_path) if processor_path.exists() else None
    facade = Qwen35LanguageModel(
        model, tokenizer, processor, stop=stop_tokens(directory, model.config.eos)
    )
    return CompositeModel(facade, _chat(directory, facade))


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    Qwen35Config,
    Qwen35,
    weights,
    _composite,
    model_types=("qwen3_5", "qwen3_5_moe"),
)
