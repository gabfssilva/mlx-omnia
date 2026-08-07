"""Qwen3-VL and Qwen3-VL-MoE — **the text trunk only**.

Authoritative semantics: transformers' modeling_qwen3_vl.py; the reference port
covers the same subset.

**The vision tower is not ported.** These checkpoints carry a ViT under `model.visual.*`
(or `visual.*`), and this file drops it at load. A Qwen3-VL loaded through here answers
text and cannot see an image — `qwen3_5/vision.py` is what a real vision port looks like
in this repo, and this is not that. It is here so the backbone is reachable and so the
text half is exercised; the tower is the work that remains.

Given that, the trunk *is* Qwen3 (dense) or Qwen3-MoE (sparse): same block, same q/k
norm, same router. So no tree is declared here. What this file owns is the load-time
rename: the VL checkpoint nests the language model one level deeper
(`model.language_model.layers.*` against `model.layers.*`), and the MoE variant ships its
experts as one `[E, in, 2·inner]` `gate_up_proj` — transposed relative to the
`[E, out, in]` layout `SwitchLinear` declares, and split rather than interleaved.

`rope_scaling.mrope_section` is ignored, as in the reference port: multimodal RoPE only differs from
the plain table once image positions enter, and no image can enter here.
"""

from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    split_stacked_gate_up,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.qwen3.config import Qwen3VLConfig, Qwen3VLMoEConfig
from sideros.models.qwen3.model import Qwen3, Qwen3MoE

NESTED = "model.language_model."
TOWERS = ("model.visual.", "visual.", "model.vision_tower.", "vision_tower.")


def flatten_language_model(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Drop the tower and lift the text trunk back to `model.*`."""
    lifted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith(TOWERS):
            continue
        lifted[key.replace(NESTED, "model.", 1) if key.startswith(NESTED) else key] = value
    return lifted


def dense_weights(
    directory: Path, config: Qwen3VLConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    text = config.dense
    layers = text.num_hidden_layers
    return prepare_weights(
        text,
        load_shards(directory),
        [flatten_language_model, lambda weights: fuse_qkv(weights, layers)],
        dtype,
    )


def moe_weights(
    directory: Path, config: Qwen3VLMoEConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    text = config.moe
    layers = text.num_hidden_layers
    return prepare_weights(
        text,
        load_shards(directory),
        [
            flatten_language_model,
            lambda weights: fuse_qkv(weights, layers),
            lambda weights: split_stacked_gate_up(weights, layers),
            lambda weights: interleave_gate_up(weights, layers),
        ],
        dtype,
    )


def _dense_tree(config: Qwen3VLConfig) -> Qwen3:
    return Qwen3(config.dense)


def _moe_tree(config: Qwen3VLMoEConfig) -> Qwen3MoE:
    return Qwen3MoE(config.moe)


def _dense_composite(directory: Path, model: Qwen3) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos),
        ),
        chat_capabilities(directory),
    )


def _moe_composite(directory: Path, model: Qwen3MoE) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos),
        ),
        chat_capabilities(directory),
    )


_FILES = (
    "config.json",
    "model*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

CHECKPOINT = checkpoint(
    _FILES,
    Qwen3VLConfig,
    _dense_tree,
    dense_weights,
    _dense_composite,
    model_types=("qwen3_vl",),
)
MOE_CHECKPOINT = checkpoint(
    _FILES,
    Qwen3VLMoEConfig,
    _moe_tree,
    moe_weights,
    _moe_composite,
    model_types=("qwen3_vl_moe",),
)
