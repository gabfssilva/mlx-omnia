from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    concat_gate_up,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stack_experts,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.engine.models.bailing_hybrid.model import BailingHybrid


def weights(
    directory: Path, config: BailingHybridConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            lambda weights: _drop_mtp(weights, layers),
            lambda weights: _stack_convs(weights, config),
            lambda weights: _fuse_kda_proj(weights, config),
            lambda weights: _split_kv_b(weights, config),
            lambda weights: _rename_router(weights, layers),
            lambda weights: stack_experts(weights, layers, config.num_experts),
            lambda weights: interleave_gate_up(weights, layers),
            lambda weights: concat_gate_up(weights, layers),
        ],
        dtype,
    )


_KDA_PROJECTIONS = ("q", "k", "v", "f", "g", "b")
_PACKED = ("weight", "scales", "biases")


def _drop_mtp(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`num_nextn_predict_layers` blocks past the trunk, plus the norms that only the MTP
    head uses. Addressed by index: the MTP block carries the same leaf names as a real
    layer."""
    return {
        key: value
        for key, value in weights.items()
        if not (
            key.startswith("model.layers.")
            and key.split(".")[2].isdigit()
            and int(key.split(".")[2]) >= layers
        )
    }


def _stack_convs(
    weights: dict[str, mx.array], config: BailingHybridConfig
) -> dict[str, mx.array]:
    """The three depthwise convs concatenated on the channel axis, `[3·width, kernel]`.
    They are never quantized and they share the kernel size, so the concatenation is the
    same reorder the fused projection is — and it is what lets the step read the taps
    without stacking three leaves on every token."""
    for layer, attends in enumerate(config.attends):
        prefix = f"model.layers.{layer}.attention."
        keys = [f"{prefix}{name}_conv1d.weight" for name in ("q", "k", "v")]
        if attends or not all(key in weights for key in keys):
            continue
        taps = [weights.pop(key) for key in keys]
        weights[f"{prefix}conv1d.weight"] = mx.concatenate(
            [tap.reshape(tap.shape[0], -1) for tap in taps], axis=0
        )
    return weights


def _fuse_kda_proj(
    weights: dict[str, mx.array], config: BailingHybridConfig
) -> dict[str, mx.array]:
    """q‖k‖v‖f‖g‖b concatenated on the output axis: they read the same input, so one
    matmul does what six did.

    Row-aligned, therefore bit-exact — every output row keeps the quantization groups it
    was packed with — and worth the trouble because the shape is what the one-token read
    is limited by, not the dispatch count: the six leaves of a KDA layer move 0.975 GB at
    235 GB/s and the fused matrix moves the same bytes at 405 GB/s (`docs/models/
    bailing_hybrid.md`). A checkpoint that quantizes them apart has no common matrix; the
    six oQ mixes of Ling 3.0 all agree within a layer, and one that does not is rejected
    rather than silently requantized."""
    for layer, attends in enumerate(config.attends):
        prefix = f"model.layers.{layer}.attention."
        leaves = [f"{prefix}{name}_proj" for name in _KDA_PROJECTIONS]
        if attends or not all(f"{leaf}.weight" in weights for leaf in leaves):
            continue
        for suffix in _PACKED:
            present = [f"{leaf}.{suffix}" for leaf in leaves]
            if not any(key in weights for key in present):
                continue
            formats = {
                (weights[key].shape[1:], weights[key].dtype)
                for key in present
                if key in weights
            }
            if len(formats) != 1 or not all(key in weights for key in present):
                raise ValueError(
                    f"bailing_hybrid: layer {layer} carries {suffix} in more than one"
                    f" format across {', '.join(_KDA_PROJECTIONS)}_proj, which the fused"
                    " projection cannot hold"
                )
            weights[f"{prefix}fused_proj.{suffix}"] = mx.concatenate(
                [weights.pop(key) for key in present], axis=0
            )
    return weights


def _split_kv_b(
    weights: dict[str, mx.array], config: BailingHybridConfig
) -> dict[str, mx.array]:
    """`kv_b_proj` is `[heads · (nope + v), kv_lora]` — two different products sharing one
    matrix. Split per head at load: the key half transposed into `embed_q` (the query is
    what moves through it on the absorbed step), the value half as `unembed_out`. A
    checkpoint converted by the reference converter already carries both and skips this."""
    nope, v_dim = config.qk_nope_head_dim, config.v_head_dim
    for layer, attends in enumerate(config.attends):
        key = f"model.layers.{layer}.attention.kv_b_proj.weight"
        if not attends or key not in weights:
            continue
        stacked = weights.pop(key).reshape(config.num_attention_heads, nope + v_dim, -1)
        leaf = f"model.layers.{layer}.attention."
        weights[f"{leaf}embed_q.weight"] = mx.contiguous(
            mx.swapaxes(stacked[:, :nope], -1, -2)
        )
        weights[f"{leaf}unembed_out.weight"] = mx.contiguous(stacked[:, nope:])
    return weights


def _rename_router(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """The HF checkpoint keeps the router as a bare matrix under `mlp.gate.weight`; the
    mlx conversions wrap it in the `gate_proj` leaf this tree declares — and quantize it,
    which a bare parameter cannot carry. Both arrive here."""
    for layer in range(layers):
        leaf = f"model.layers.{layer}.mlp.gate."
        if f"{leaf}weight" in weights:
            weights[f"{leaf}gate_proj.weight"] = weights.pop(f"{leaf}weight")
    return weights


def _composite(directory: Path, model: BailingHybrid) -> LanguageModel[ModelInput]:
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
    BailingHybridConfig,
    BailingHybrid,
    weights,
    _composite,
    model_types=("bailing_hybrid",),
)
