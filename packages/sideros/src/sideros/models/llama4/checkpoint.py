from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.llama4.config import Llama4Config
from sideros.models.llama4.model import Llama4


def weights(directory: Path, config: Llama4Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    layers = config.text_config.num_hidden_layers
    renamed = _rename(load_shards(directory))
    return prepare_weights(
        config,
        renamed,
        [
            lambda w: _prepare_experts(w, layers),
            lambda w: fuse_qkv(w, layers),
            lambda w: interleave_gate_up(w, layers),
            lambda w: concat_gate_up(w, layers),
            lambda w: _fuse_shared_expert(w, layers),
        ],
        dtype,
    )


def _rename(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Strip `language_model.` prefix (the Llama4 checkpoint nests the trunk) and
    rename `feed_forward` → `mlp` (Sideros convention)."""
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        key = key.removeprefix("language_model.")
        key = key.replace(".feed_forward.", ".mlp.")
        renamed[key] = value
    return renamed


def _prepare_experts(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Rename `mlp.experts.*` → `mlp.switch_mlp.*` and `mlp.router` → `mlp.gate`.

    The original meta-llama checkpoint ships `experts.gate_up_proj` as a single
    `[E, hidden, 2*inner]` nn.Parameter and `experts.down_proj` as `[E, inner,
    hidden]` — both need a `swapaxes(1, 2)` to reach the SwitchLinear `[E, out, in]`
    layout. The mlx-community (sanitized) conversion already splits and transposes
    them into `gate_proj.weight` / `up_proj.weight` / `down_proj.weight` — only the
    rename is needed. Both cases are handled.
    """
    for layer in range(layers):
        ep = f"model.layers.{layer}.mlp.experts."
        sp = f"model.layers.{layer}.mlp.switch_mlp."

        # Non-sanitized: single gate_up_proj parameter [E, hidden, 2*inner]
        key = f"{ep}gate_up_proj"
        if key in weights:
            v = weights.pop(key)
            gate, up = mx.split(v, 2, axis=-1)
            weights[f"{sp}gate_proj.weight"] = mx.swapaxes(gate, 1, 2)
            weights[f"{sp}up_proj.weight"] = mx.swapaxes(up, 1, 2)
            mx.eval(weights[f"{sp}gate_proj.weight"], weights[f"{sp}up_proj.weight"])

        # Non-sanitized: single down_proj parameter [E, inner, hidden]
        key = f"{ep}down_proj"
        if key in weights:
            v = weights.pop(key)
            weights[f"{sp}down_proj.weight"] = mx.swapaxes(v, 1, 2)
            mx.eval(weights[f"{sp}down_proj.weight"])

        # Sanitized: already split + transposed, just rename
        for suffix in ("weight", "scales", "biases"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                old = f"{ep}{proj}.{suffix}"
                if old in weights:
                    weights[f"{sp}{proj}.{suffix}"] = weights.pop(old)

    # Router → gate
    for layer in range(layers):
        old = f"model.layers.{layer}.mlp.router.weight"
        if old in weights:
            weights[f"model.layers.{layer}.mlp.gate.weight"] = weights.pop(old)

    return weights


def _fuse_shared_expert(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Concatenate shared_expert.gate_proj and .up_proj on the output axis."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.shared_expert."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _composite(directory: Path, model: Llama4) -> LanguageModel[ModelInput]:
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            "Llama4 uses a tiktoken/o200k tokenizer; sideros has no tiktoken reader yet. "
            "The model loads and forwards but cannot serve text until a tokenizer reader is added."
        )
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(tokenizer_path),
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
    Llama4Config,
    Llama4,
    weights,
    _composite,
    model_types=("llama4",),
)
