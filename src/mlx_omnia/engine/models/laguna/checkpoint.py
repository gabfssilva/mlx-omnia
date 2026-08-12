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
from mlx_omnia.engine.core.mxcompat import quantize
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.laguna.config import LagunaConfig
from mlx_omnia.engine.models.laguna.model import Laguna


def weights(directory: Path, config: LagunaConfig, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    """e_score_correction_bias is float32 whatever the checkpoint's precision: the router
    adds it in float32, and a conversion that wrote it in the model's dtype (the oQ
    quantizations do) is upcast back rather than followed."""
    loaded = _rename_vlm(load_shards(directory))
    reject_dtype_cast(dtype, loaded)

    loaded = {
        key: value.astype(mx.float32)
        if key.endswith("e_score_correction_bias")
        else (value if dtype is None else value.astype(dtype))
        for key, value in loaded.items()
    }

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    loaded = concat_gate_up(loaded, config.num_hidden_layers)
    loaded = _stack_experts(loaded, config)
    loaded = _interleave_stacked_experts(loaded, config)
    loaded = _fuse_shared(loaded, config)
    return _requantize_dense(loaded, config)


def _rename_vlm(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """The mlx-vlm conversion nests the trunk under `language_model.` and the router
    under `gate.proj`, with the selection bias as the router's sibling. The tree keeps
    the original checkpoint's names; pure renames, no arithmetic."""
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        key = key.removeprefix("language_model.")
        key = key.replace(".mlp.gate.proj.", ".mlp.gate.")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
        renamed[key] = value
    return renamed


def _stack_experts(weights: dict[str, mx.array], config: LagunaConfig) -> dict[str, mx.array]:
    """Per-expert gate/up/down stacked into SwitchLinear leaves, gate‖up
    row-interleaved; e_score_correction_bias moves from experts.* to mlp.*."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp."

        bias_key = f"{prefix}experts.e_score_correction_bias"
        if bias_key in weights:
            weights[f"{prefix}e_score_correction_bias"] = weights.pop(bias_key)

        for suffix in ("weight", "scales", "biases"):
            gate_keys = [
                f"{prefix}experts.{e}.gate_proj.{suffix}" for e in range(config.num_experts)
            ]
            up_keys = [f"{prefix}experts.{e}.up_proj.{suffix}" for e in range(config.num_experts)]
            if not all(key in weights for key in gate_keys + up_keys):
                continue
            gates = mx.stack([weights.pop(key) for key in gate_keys])
            ups = mx.stack([weights.pop(key) for key in up_keys])
            interleaved = mx.stack([gates, ups], axis=2).reshape(
                config.num_experts, 2 * config.moe_intermediate_size, -1
            )
            mx.eval(interleaved)
            weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = interleaved

        for suffix in ("weight", "scales", "biases"):
            down_keys = [
                f"{prefix}experts.{e}.down_proj.{suffix}" for e in range(config.num_experts)
            ]
            if not all(key in weights for key in down_keys):
                continue
            stacked = mx.stack([weights.pop(key) for key in down_keys])
            mx.eval(stacked)
            weights[f"{prefix}switch_mlp.down_proj.{suffix}"] = stacked

    return weights


def _interleave_stacked_experts(
    weights: dict[str, mx.array], config: LagunaConfig
) -> dict[str, mx.array]:
    """A conversion that already stacked the experts ships `switch_mlp.gate_proj` and
    `switch_mlp.up_proj` as `[E, inner, ·]`; only the row interleave is left. Same row
    rule as `_stack_experts`, holding for packed weight, scales and biases."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp.switch_mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            gates, ups = (weights.pop(key) for key in keys)
            interleaved = mx.stack([gates, ups], axis=2).reshape(
                config.num_experts, 2 * config.moe_intermediate_size, -1
            )
            mx.eval(interleaved)
            weights[f"{prefix}gate_up_proj.{suffix}"] = interleaved
    return weights


def _fuse_shared(weights: dict[str, mx.array], config: LagunaConfig) -> dict[str, mx.array]:
    """Shared expert's gate‖up concatenated on the output axis, like the dense MLP."""
    for layer in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer] != "sparse":
            continue
        prefix = f"model.layers.{layer}.mlp.shared_expert."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


DENSE_GROUP_SIZE = 32
DENSE_BITS = 8
ATTENTION_GROUP_SIZE = 16
ATTENTION_BITS = 4


def _requantize_dense(weights: dict[str, mx.array], config: LagunaConfig) -> dict[str, mx.array]:
    """A checkpoint that ships its expert stacks quantized and everything else dense spends
    most of its per-token bandwidth on the dense side: on Laguna-XS-2.1-NVFP4 the attention
    is 2.862 and the `lm_head` 0.411 of the 4.036 GB read per token, against 0.552 GB for
    the eight routed experts. NVFP4 reduces the two main attention projections; the head
    stays dense because greedy decode prunes its exact reads with a certified bound.

    Emitting packed tensors here rather than quantizing the built tree is what keeps the
    spine untouched — `infer_quantization` reads the format off the shapes and builds the
    quantized leaf itself. A leaf that already arrived packed (every oQ conversion of
    Laguna-S) carries its own scales and is left alone. `embed_tokens` stays dense: it is a
    one-row lookup per token, so its width costs nothing per step.

    The cost is reordering, not accuracy. For attention, measured: the perturbation moves
    the logits 3.200e-01 where the model's own prefill-vs-stepwise batching noise already
    moves them 2.633e-01, with the same tie-gap distribution (median 0.250) — 1.22x a floor
    the architecture carries on its own, well inside the 3x the suite requires. The routing
    near-ties cascading through 40 sparse layers dominate; the weights round-trip at 5e-03.
    `docs/models/laguna.md` carries the numbers.
    """
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.self_attn."
        for name in ("qkv_proj", "o_proj"):
            _pack(
                weights,
                f"{prefix}{name}",
                group_size=ATTENTION_GROUP_SIZE,
                bits=ATTENTION_BITS,
                mode="nvfp4",
            )
        _pack(weights, f"{prefix}g_proj")
    return weights


def _pack(
    weights: dict[str, mx.array],
    path: str,
    *,
    group_size: int = DENSE_GROUP_SIZE,
    bits: int = DENSE_BITS,
    mode: str = "affine",
) -> None:
    weight = weights.get(f"{path}.weight")
    if weight is None or f"{path}.scales" in weights:
        return
    packed, scales, *rest = quantize(weight, group_size=group_size, bits=bits, mode=mode)
    mx.eval(packed, scales, *rest)
    weights[f"{path}.weight"] = packed
    weights[f"{path}.scales"] = scales
    if rest:
        weights[f"{path}.biases"] = rest[0]


def _composite(directory: Path, model: Laguna) -> LanguageModel[ModelInput]:
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
    LagunaConfig,
    Laguna,
    weights,
    _composite,
    model_types=("laguna",),
)
