import math
from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import (
    ChatCapability,
    MultimodalChatCapability,
    chat_template,
)
from mlx_omnia.engine.checkpoint import (
    ImageCost,
    Sight,
    checkpoint,
    concat_gate_up,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    materialize,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.engine.core.config import load_config
from mlx_omnia.engine.language import LanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.step3p7.config import Step3p7Config
from mlx_omnia.engine.models.step3p7.model import Step3p7, Step3p7LanguageModel, sees
from mlx_omnia.engine.processors.step3p7 import (
    TILE_GRID,
    WINDOW_SIZE,
    Step3p7Processor,
)


def weights(
    directory: Path, config: Step3p7Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    text = config.text_config
    loaded = _sanitize(load_shards(directory), config)
    reject_dtype_cast(dtype, loaded)

    if dtype is not None:
        loaded = {
            key: value
            if key.endswith("router_bias")
            else value.astype(dtype)
            for key, value in loaded.items()
        }

    if config.tied:
        drop_tied_head(loaded)

    loaded = fuse_qkv(loaded, text.num_hidden_layers)
    loaded = concat_gate_up(loaded, text.num_hidden_layers)
    loaded = _interleave_moe_gate_up(loaded, config)
    loaded = _fuse_shared_expert(loaded, config)
    loaded = _add_norm_offset(loaded)
    loaded = _fold_conv1(loaded)
    return loaded


_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "norm.weight",
)


def _interleave_moe_gate_up(
    weights: dict[str, mx.array], config: Step3p7Config
) -> dict[str, mx.array]:
    for layer in config.text_config.moe_layers:
        prefix = f"model.layers.{layer}.moe."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            parts = [weights.pop(key) for key in keys]
            experts, rows, cols = parts[0].shape
            interleaved = mx.stack(parts, axis=2).reshape(experts, 2 * rows, cols)
            materialize(interleaved)
            weights[f"{prefix}gate_up_proj.{suffix}"] = interleaved
    return weights


def _fuse_shared_expert(
    weights: dict[str, mx.array], config: Step3p7Config
) -> dict[str, mx.array]:
    for layer in config.text_config.moe_layers:
        prefix = f"model.layers.{layer}.share_expert."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            materialize(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _add_norm_offset(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {
        key: (value + 1 if key.endswith(_NORM_SUFFIXES) else value)
        for key, value in weights.items()
    }


def _sanitize(weights: dict[str, mx.array], config: Step3p7Config) -> dict[str, mx.array]:
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        if ".mtp" in key or key.startswith("mtp."):
            continue
        if key.startswith("model.layers."):
            layer_num = int(key.split(".")[2])
            if layer_num >= config.text_config.num_hidden_layers:
                continue
        renamed[key] = value
    return renamed


def _fold_conv1(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Reshape the vision tower's Conv2d patch embed weight from [out, 3, P, P]
    to the folded [out, 3*P*P] matmul layout the tree declares."""
    key = "vision_model.conv1.weight"
    if key in weights and weights[key].ndim == 4:
        weights[key] = weights[key].reshape(weights[key].shape[0], -1)
    return weights


def _chat(
    directory: Path, facade: Step3p7LanguageModel
) -> list[ChatCapability | MultimodalChatCapability]:
    template = chat_template(directory)
    if template is None:
        return []
    marker = facade.image_marker
    if marker is None:
        return [ChatCapability(template)]
    return [MultimodalChatCapability(template, marker)]


def _composite(directory: Path, model: Step3p7) -> LanguageModel[ModelInput]:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    processor = Step3p7Processor.from_directory(directory, model.config)
    facade = Step3p7LanguageModel(
        model, tokenizer, processor, stop=stop_tokens(directory, model.config.eos)
    )
    return CompositeModel(facade, _chat(directory, facade))


_TYPES = ("step3p7",)


def image_rows(processor: Step3p7Processor, height: int, width: int) -> int:
    """What `assemble_ids` would lay down for one image this size: a `<patch_start>`, the
    tile's grid with a newline between its rows and a `<patch_end>` per tile, then the base
    image's grid between `<im_start>` and `<im_end>`.

    Written against that method and testable against it, which is the only reason it is not
    inside `_sight`: an arithmetic a test restates is an arithmetic no test protects."""
    tiles = math.ceil(height / WINDOW_SIZE) * math.ceil(width / WINDOW_SIZE)
    newlines = TILE_GRID - 1 if processor.patch_newline_id >= 0 else 0
    per_tile = 2 + TILE_GRID * TILE_GRID + newlines
    return tiles * per_tile + 2 + processor.image_token_len


def _sight(directory: Path) -> Sight | None:
    """This family does not resize: the tiles are cut out of the image as it came, and the
    base is a square pad of it. What an image costs is therefore the tiling — which is why
    the rows come off `assemble_ids` and not off a patch grid."""
    config = load_config(Step3p7Config, directory / "config.json", allowed_model_types=_TYPES)
    processor = Step3p7Processor.from_directory(directory, config)
    if processor is None or not sees(config, processor):
        return None

    def cost(height: int, width: int) -> ImageCost:
        return ImageCost(height, width, image_rows(processor, height, width))

    return cost


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    ),
    Step3p7Config,
    Step3p7,
    weights,
    _composite,
    model_types=_TYPES,
    sight=_sight,
)
