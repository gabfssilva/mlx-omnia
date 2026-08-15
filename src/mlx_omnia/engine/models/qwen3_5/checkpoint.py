from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import (
    ChatCapability,
    MultimodalChatCapability,
    chat_template,
)
from mlx_omnia.engine.checkpoint import (
    MTP_PREFIX,
    Drafter,
    ImageCost,
    Pending,
    Sight,
    attach_weights,
    checkpoint,
    declared_plan,
    drop_tied_head,
    fusible,
    load_shards,
    materialize,
    reject_dtype_cast,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.engine.core.config import load_config
from mlx_omnia.engine.language import LanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.qwen3_5.config import Qwen35Config, Qwen35TextConfig
from mlx_omnia.engine.models.qwen3_5.model import Qwen35, Qwen35LanguageModel, sees
from mlx_omnia.engine.models.qwen3_5.mtp import Qwen35MTP
from mlx_omnia.engine.models.qwen3_5.vision import (
    Grid,
    ProcessorConfig,
    load_processor_config,
    normalized_patch_weight,
    smart_resize,
)
from mlx_omnia.engine.quant.quantization import QuantizationPlan


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
            loaded[renamed] = array.astype(dtype) if cast else array

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

    loaded = stack_experts(loaded, text.num_hidden_layers, text.num_experts)
    loaded = _fuse_projections(loaded, text)
    return _fuse_moe(loaded, text)


_ZERO_CENTERED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)

_MTP_ZERO_CENTERED = (
    *_ZERO_CENTERED,
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def load_mtp(directory: Path, dtype: mx.Dtype | None = None) -> Qwen35MTP:
    """The MTP head of a Qwen3.8 checkpoint, as a tree of its own.

    A directory that is also the model's: `mtp.*` shares the target's shards, and the
    trunk's `_renamed` drops it. It does not go through `mlx_omnia.load` — it serves no
    task, has no tokenizer, and its logits are the target's `lm_head` over what it returns.
    """
    config = load_config(Qwen35Config, directory / "config.json", allowed_model_types=_TYPES)
    return attach_weights(
        Qwen35MTP(config.text_config), mtp_weights(directory, config, dtype),
        declared=mtp_plan(directory),
    )


def mtp_plan(directory: Path) -> QuantizationPlan | None:
    """The head's own leaves out of the entry's declaration, with the prefix off, or `None`
    for a source checkpoint that declares nothing."""
    declared = declared_plan(directory / "config.json")
    if declared is None:
        return None
    return {
        leaf.removeprefix(MTP_PREFIX): format
        for leaf, format in declared.items()
        if leaf.startswith(MTP_PREFIX)
    }


def mtp_weights(
    directory: Path, config: Qwen35Config, dtype: mx.Dtype | None = None
) -> dict[str, mx.array]:
    """The head's tensors in the tree's names.

    The zero-centering probe is the trunk's own — the conv layout — read off the same
    shards: the head only exists in two dialects, raw HF (torch conv `[dim, 1, K]`, every
    norm still `1 + w`) and an entry this engine wrote (conv squeezed, shift already
    folded), because the mlx conversions drop `mtp.*` on the floor.
    """
    shards = load_shards(directory)
    loaded = {
        key.removeprefix(MTP_PREFIX): value
        for key, value in shards.items()
        if key.startswith(MTP_PREFIX)
    }
    if not loaded:
        raise ValueError(f"{directory} carries no MTP head (`{MTP_PREFIX}*`)")
    reject_dtype_cast(dtype, loaded)
    if dtype is not None:
        loaded = {key: value.astype(dtype) for key, value in loaded.items()}

    text = config.text_config
    first_linear = text.layer_types.index("linear_attention")
    probe = f"layers.{first_linear}.linear_attn.conv1d.weight"
    conv = next(value for key, value in shards.items() if key.endswith(probe))
    if conv.shape[1] == 1:
        for key, value in loaded.items():
            if key == "norm.weight" or key.endswith(_MTP_ZERO_CENTERED):
                loaded[key] = value + 1

    _fuse(loaded, "layers.0.self_attn.", "fused_proj", ("q_proj", "k_proj", "v_proj"))
    _fuse(loaded, "layers.0.mlp.", "gate_up_proj", ("gate_proj", "up_proj"))
    return loaded


def mtp_pending(directory: Path, dtype: mx.Dtype | None) -> Pending[Qwen35MTP]:
    """The head's own split for the quantizer: a lazy tree to resolve a plan against, the
    tensors, and the tail — `task.write_entry` packs it under `MTP_PREFIX` beside the
    trunk."""
    config = load_config(Qwen35Config, directory / "config.json", allowed_model_types=_TYPES)
    tree = Qwen35MTP(config.text_config)
    return Pending(
        tree,
        lambda: mtp_weights(directory, config, dtype),
        lambda prepared: attach_weights(tree, prepared),
    )


MTP = Drafter(("config.json", "model*.safetensors"), load_mtp, mtp_pending)
"""The head as the quantizer sees it: a `Drafter` because that is exactly what it is — a
tree with weights to pack, no task to serve and no tokenizer."""


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


_SUFFIXES = ("weight", "scales", "biases")


def _fuse(
    weights: dict[str, mx.array], prefix: str, name: str, parts: tuple[str, ...]
) -> None:
    """The parts concatenated on the output axis — row-aligned in every representation,
    so dense and packed fuse alike. The originals leave the dict, otherwise both copies
    stay resident.

    A per-leaf plan is what breaks that: a mixed-precision quantizer picks a width per
    projection, and qkv in 3 bits next to z in 5 has no common matrix. The parts then move
    under the fused name instead of into it, numbered in the order the model splits the
    output, and `attach_weights` builds a `SegmentedLinear` over them. The checkpoint is
    untouched either way — the decision is the loader's, taken by comparing the tensors.
    """
    if not all(f"{prefix}{part}.weight" in weights for part in parts):
        return
    if not all(fusible(weights, [f"{prefix}{part}.{s}" for part in parts]) for s in _SUFFIXES):
        for index, part in enumerate(parts):
            for suffix in _SUFFIXES:
                key = f"{prefix}{part}.{suffix}"
                if key in weights:
                    weights[f"{prefix}{name}.parts.{index}.{suffix}"] = weights.pop(key)
        return
    for suffix in _SUFFIXES:
        keys = [f"{prefix}{part}.{suffix}" for part in parts]
        if not all(key in weights for key in keys):
            continue
        fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
        materialize(fused)
        weights[f"{prefix}{name}.{suffix}"] = fused


def _fuse_projections(
    weights: dict[str, mx.array], config: Qwen35TextConfig
) -> dict[str, mx.array]:
    """One fused projection per mixer instead of three or four, and one per dense MLP."""
    for layer, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            prefix, parts = f"model.layers.{layer}.self_attn.", ("q_proj", "k_proj", "v_proj")
        else:
            prefix = f"model.layers.{layer}.linear_attn."
            parts = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        _fuse(weights, prefix, "fused_proj", parts)

    for layer in range(config.num_hidden_layers):
        _fuse(weights, f"model.layers.{layer}.mlp.", "gate_up_proj", ("gate_proj", "up_proj"))
    return weights


def _interleave(weights: dict[str, mx.array], prefix: str) -> None:
    """gate and up row by row ([g0,u0,g1,…]) into one leaf — the layout the T=1 kernels
    read a pair from. Rows again, so packed weights, scales and biases take the same
    path; the stack's leading expert axis rides along untouched."""
    for suffix in _SUFFIXES:
        keys = [f"{prefix}{part}_proj.{suffix}" for part in ("gate", "up")]
        if not all(key in weights for key in keys):
            continue
        stacked = [weights.pop(key) for key in keys]
        *lead, rows, cols = stacked[0].shape
        fused = mx.stack(stacked, axis=-2).reshape(*lead, 2 * rows, cols)
        materialize(fused)
        weights[f"{prefix}gate_up_proj.{suffix}"] = fused


def _fuse_moe(weights: dict[str, mx.array], config: Qwen35TextConfig) -> dict[str, mx.array]:
    """The sparse block's load-time fusions: the shared expert's logit becomes row 256 of
    the router's matrix, so one gemv produces the routing logits and the shared gate
    together, and gate/up interleave into one leaf per stack.

    The shared expert's pair stays a leaf of its own rather than slot 256 of the routed
    stack: a stack holds one quantization format, and a per-leaf plan gives that expert
    its own. The T=1 kernels read it through their spare slot, so the fused step is the
    same four dispatches either way.
    """
    if config.num_experts == 0:
        return weights
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        _fuse(weights, prefix, "gate", ("gate", "shared_expert_gate"))
        _interleave(weights, f"{prefix}switch_mlp.")
        _interleave(weights, f"{prefix}shared_expert.")
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


_TYPES = ("qwen3_5", "qwen3_5_moe")


def _processor(directory: Path) -> ProcessorConfig | None:
    path = directory / "preprocessor_config.json"
    return load_processor_config(path) if path.exists() else None


def _composite(directory: Path, model: Qwen35) -> LanguageModel[ModelInput]:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    processor = _processor(directory)
    facade = Qwen35LanguageModel(
        model, tokenizer, processor, stop=stop_tokens(directory, model.config.eos)
    )
    return CompositeModel(facade, _chat(directory, facade))


def _sight(directory: Path) -> Sight | None:
    """The same two steps `process_image` takes before the tower reads anything: the
    resize, then the patch grid the merger folds by 2x2."""
    processor = _processor(directory)
    config = load_config(Qwen35Config, directory / "config.json", allowed_model_types=_TYPES)
    if processor is None or not sees(config, processor):
        return None

    def cost(height: int, width: int) -> ImageCost:
        read = smart_resize(height, width, processor)
        grid = Grid(1, read[0] // processor.patch_size, read[1] // processor.patch_size)
        return ImageCost(read[0], read[1], grid.tokens(processor.merge_size))

    return cost


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
    model_types=_TYPES,
    sight=_sight,
)
