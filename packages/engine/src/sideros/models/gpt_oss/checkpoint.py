from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    fuse_qkv,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.gpt_oss.config import GPTOSSConfig
from sideros.models.gpt_oss.model import GPTOSS


def weights(directory: Path, config: GPTOSSConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    loaded = _unpack_experts(load_shards(directory), config.num_hidden_layers)
    # The checkpoint is MXFP4: a cast keeps the shape and destroys the numbers. Checked
    # after the unpack, which is what renames `{proj}_scales` into the `.scales` it reads.
    reject_dtype_cast(dtype, loaded)
    return fuse_qkv(loaded, config.num_hidden_layers)


def _unpack_experts(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`*_blocks` is [E, out, groups, 16] uint8; `gather_qmm` reads MXFP4 nibbles as
    uint32 words, so the last two axes are reinterpreted and flattened — a pure view.
    The checkpoint's flat `{proj}_blocks`/`{proj}_scales` become the `{proj}.weight`/
    `{proj}.scales` of a leaf, which is what makes the format inferable like any other."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        for proj in ("gate_up_proj", "down_proj"):
            blocks = weights.pop(f"{prefix}{proj}_blocks", None)
            if blocks is None:
                continue
            experts, out = blocks.shape[0], blocks.shape[1]
            weights[f"{prefix}{proj}.weight"] = blocks.view(mx.uint32).reshape(experts, out, -1)
            weights[f"{prefix}{proj}.scales"] = weights.pop(f"{prefix}{proj}_scales")
    return weights


def _composite(directory: Path, model: GPTOSS) -> LanguageModel[ModelInput]:
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
        "generation_config.json",
        "chat_template.jinja",
    ),
    GPTOSSConfig,
    GPTOSS,
    weights,
    _composite,
    model_types=("gpt_oss",),
)
