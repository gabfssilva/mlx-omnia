from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    concat_gate_up,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.falcon_h1.config import FalconH1Config
from mlx_omnia.engine.models.falcon_h1.model import FalconH1


def weights(
    directory: Path, config: FalconH1Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """The checkpoint's tensors in the tree's names and layout, up to (not
    including) the tree itself."""
    loaded = load_shards(directory)
    reject_dtype_cast(dtype, loaded)

    if dtype is not None:
        loaded = {
            key: value
            if key.endswith("A_log")
            else value.astype(dtype)
            for key, value in loaded.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    loaded = _fold_mup(loaded, config)
    loaded = _transpose_conv1d(loaded, config)
    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    return concat_gate_up(loaded, config.num_hidden_layers)


def _compute_mup_vector(config: FalconH1Config) -> mx.array:
    """The per-row μP vector for ``in_proj``: each chunk of the output is scaled
    by its ``ssm_multipliers`` entry — ``[intermediate, intermediate, G*Ds, G*Ds,
    H]``."""
    sizes = [
        config.mamba_d_ssm,
        config.mamba_d_ssm,
        config.mamba_n_groups * config.mamba_d_state,
        config.mamba_n_groups * config.mamba_d_state,
        config.mamba_n_heads,
    ]
    return mx.concatenate(
        [
            mx.broadcast_to(mx.array(m), (s,))
            for s, m in zip(sizes, config.ssm_multipliers, strict=True)
        ]
    )


def _fold_mup(weights: dict[str, mx.array], config: FalconH1Config) -> dict[str, mx.array]:
    """Fold the nine μP multipliers and the per-row ssm vector into the weights
    at load (equivalent to transformers' runtime multiply for inference, modulo
    dtype). A_log and conv1d are untouched."""
    mup_vector = _compute_mup_vector(config)
    for name, param in list(weights.items()):
        if name.endswith("embed_tokens.weight"):
            weights[name] = param * config.embedding_multiplier
        elif name.endswith("lm_head.weight"):
            weights[name] = param * config.lm_head_multiplier
        elif (
            name.endswith("self_attn.q_proj.weight")
            or name.endswith("self_attn.k_proj.weight")
            or name.endswith("self_attn.v_proj.weight")
        ):
            # attention_in_multiplier scales the input before q/k/v (transformers applies
            # it to h before the projections); key_multiplier additionally scales k.
            mult = config.attention_in_multiplier
            if name.endswith("k_proj.weight"):
                mult *= config.key_multiplier
            weights[name] = param * mult
        elif name.endswith("self_attn.o_proj.weight"):
            weights[name] = param * config.attention_out_multiplier
        elif name.endswith("mamba.out_proj.weight"):
            weights[name] = param * config.ssm_out_multiplier
        elif name.endswith("mlp.gate_proj.weight"):
            weights[name] = param * config.mlp_multipliers[0]
        elif name.endswith("mlp.down_proj.weight"):
            weights[name] = param * config.mlp_multipliers[1]
        elif name.endswith("mamba.in_proj.weight"):
            weights[name] = param * (
                config.ssm_in_multiplier * mup_vector.astype(param.dtype)[:, None]
            )
    return weights


def _transpose_conv1d(weights: dict[str, mx.array], config: FalconH1Config) -> dict[str, mx.array]:
    """torch ships conv1d as ``[conv_dim, 1, kernel]``; the house layout is
    ``[conv_dim, kernel]`` (squeezed). Apply the transform the reference
    implementation's sanitize does: ``transpose(0, 2, 1)`` then squeeze the middle axis."""
    for name, param in list(weights.items()):
        if "conv1d.weight" in name and param.ndim == 3:
            weights[name] = param.transpose(0, 2, 1).squeeze(1)
    return weights


def _composite(directory: Path, model: FalconH1) -> LanguageModel[ModelInput]:
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
    FalconH1Config,
    FalconH1,
    weights,
    _composite,
    model_types=("falcon_h1",),
)
