import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import ChatCapability, MultimodalChatCapability, chat_template
from mlx_omnia.engine.checkpoint import (
    Drafter,
    ImageCost,
    Pending,
    Sight,
    attach_weights,
    checkpoint,
    concat_gate_up,
    declared_plan,
    load_shards,
    materialize,
    prepare_weights,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.engine.core.config import load_config
from mlx_omnia.engine.language import LanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.muse_glimmer.config import (
    MuseGlimmerAssistantConfig,
    MuseGlimmerConfig,
    MuseGlimmerRoPE,
)
from mlx_omnia.engine.models.muse_glimmer.dflash import MuseGlimmerAssistant
from mlx_omnia.engine.models.muse_glimmer.model import (
    MuseGlimmer,
    MuseGlimmerLanguageModel,
    sees,
)
from mlx_omnia.engine.models.muse_glimmer.vision import (
    Grid,
    ProcessorConfig,
    load_processor_config,
    smart_resize,
)

_TYPES = ("muse_glimmer",)

_RENAMES = (
    ("model.language_model.", "model."),
    ("model.vision_tower.", "visual."),
    ("model.vision_adapter.", "vision_adapter."),
    ("model.vision_projection.", "vision_projection."),
)


def weights(
    directory: Path, config: MuseGlimmerConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.text_config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            _rename,
            lambda weights: concat_gate_up(weights, layers),
            _fold_block_norms,
        ],
        dtype,
    )


def _rename(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """The text trunk back to `model.*`, the tower and its projections to the tree's
    names."""
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        for prefix, replacement in _RENAMES:
            if key.startswith(prefix):
                key = replacement + key.removeprefix(prefix)
                break
        renamed[key] = value
    return renamed


def _fold_block_norms(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Only the four block norms are centred (`scale = 1 + w`); `model.norm` multiplies
    its weight directly, the qk norm ships no weight, and the tower's LayerNorms are
    plain — so the shared `fold_norm_scales` would fold too much."""
    for key, value in weights.items():
        if key.endswith("layernorm.weight"):
            weights[key] = value + 1
    materialize(list(weights.values()))
    return weights


def _chat(
    directory: Path, facade: MuseGlimmerLanguageModel
) -> list[ChatCapability | MultimodalChatCapability]:
    template = chat_template(directory)
    if template is None:
        return []
    marker = facade.image_marker
    if marker is None:
        return [ChatCapability(template)]
    return [MultimodalChatCapability(template, marker)]


def _processor(directory: Path) -> ProcessorConfig | None:
    path = directory / "processor_config.json"
    return load_processor_config(path) if path.exists() else None


def _composite(directory: Path, model: MuseGlimmer) -> LanguageModel[ModelInput]:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    processor = _processor(directory)
    facade = MuseGlimmerLanguageModel(
        model, tokenizer, processor, stop=stop_tokens(directory, model.config.eos)
    )
    return CompositeModel(facade, _chat(directory, facade))


def _sight(directory: Path) -> Sight | None:
    """The two steps `process_image` takes before the tower reads anything. The resize is
    this family's own — the merged-patch grid closest to the aspect ratio *under a token
    cap*, not qwen's area window — so the cap is the number that decides here."""
    processor = _processor(directory)
    config = load_config(MuseGlimmerConfig, directory / "config.json", allowed_model_types=_TYPES)
    if processor is None or not sees(config, processor):
        return None

    def cost(height: int, width: int) -> ImageCost:
        read = smart_resize(
            height,
            width,
            processor.patch_size * processor.merge_size,
            processor.max_image_tokens,
        )
        grid = Grid(1, read[0] // processor.patch_size, read[1] // processor.patch_size)
        return ImageCost(read[0], read[1], grid.tokens(processor.merge_size))

    return cost


def assistant_weights(
    directory: Path, config: MuseGlimmerAssistantConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """The drafter's shards in the tree's names. Not `prepare_weights`: there is no tied
    head to drop and no `_SpineConfig` to hand it — what the two share is the refusal to
    cast packed tensors, which is what makes a quantized entry loadable by the same path."""
    tensors = load_shards(directory)
    reject_dtype_cast(dtype, tensors)
    if dtype is not None:
        tensors = {key: value.astype(dtype) for key, value in tensors.items()}
    return _concat_assistant_gate_up(tensors, config.num_hidden_layers)


def load_assistant(directory: Path, dtype: mx.Dtype | None = None) -> MuseGlimmerAssistant:
    """The drafter, loaded on its own. It is not a `Checkpoint`: it serves no task, has no
    tokenizer and no head, and only exists bound to a target — so it does not go through
    `mlx_omnia.load`, which is the door for models that answer.

    The tail is the shared one, so which leaves are quantized comes off the tensors: a
    drafter packed by `POST /admin/quantizations` loads here with nothing else to say."""
    parsed = assistant_config(json.loads((directory / "config.json").read_text()))
    declared = declared_plan(directory / "config.json")
    if declared is None:
        weights = assistant_weights(directory, parsed, dtype)
    else:
        # An entry `save_quantized` wrote: its tensors are already a prepared dict, so the
        # gate‖up fusion is not run a second time over them.
        weights = load_shards(directory)
        reject_dtype_cast(dtype, weights)
    return attach_weights(MuseGlimmerAssistant(parsed), weights, declared=declared)


def _assistant_pending(
    directory: Path, dtype: mx.Dtype | None
) -> Pending[MuseGlimmerAssistant]:
    parsed = assistant_config(json.loads((directory / "config.json").read_text()))
    tree = MuseGlimmerAssistant(parsed)
    return Pending(
        tree,
        lambda: assistant_weights(directory, parsed, dtype),
        lambda prepared: attach_weights(tree, prepared),
    )


ASSISTANT = Drafter(("config.json", "model*.safetensors"), load_assistant, _assistant_pending)
"""The drafter carries no tokenizer and no chat template — the pair borrows the target's —
so the entry a quantization writes is the config and the weights and nothing else."""


def assistant_config(config: Mapping[str, Any]) -> MuseGlimmerAssistantConfig:
    rope = config["rope_parameters"]
    return MuseGlimmerAssistantConfig(
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        rms_norm_eps=config["rms_norm_eps"],
        sliding_window=config["sliding_window"],
        layer_types=tuple(config["layer_types"]),
        rope_parameters=MuseGlimmerRoPE(rope["rope_theta"], rope.get("rope_type", "default")),
        block_size=config["block_size"],
        mask_token_id=config["mask_token_id"],
        target_layer_ids=tuple(config["target_layer_ids"]),
    )


def _concat_assistant_gate_up(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`concat_gate_up` addresses `model.layers.*`; the drafter's trunk has no `model.`
    above it — its blocks are the tree's root."""
    for layer in range(layers):
        leaf = f"layers.{layer}.mlp."
        weights[f"{leaf}gate_up_proj.weight"] = mx.concatenate(
            [weights.pop(f"{leaf}{name}_proj.weight") for name in ("gate", "up")], axis=0
        )
    return weights


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "processor_config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ),
    MuseGlimmerConfig,
    MuseGlimmer,
    weights,
    _composite,
    model_types=_TYPES,
    sight=_sight,
)
