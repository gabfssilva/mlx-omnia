from pathlib import Path

import mlx.core as mx

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput
from sideros.models.deepseek_v4.config import DeepseekV4Config
from sideros.models.deepseek_v4.model import DeepseekV4


def weights(
    directory: Path, config: DeepseekV4Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    layers = config.num_hidden_layers
    return prepare_weights(
        config,
        load_shards(directory),
        [
            _drop_mtp,
            lambda tensors: _group_output_lora(
                tensors, layers, config.o_groups, config.o_lora_rank
            ),
            lambda tensors: _narrow_hash_tables(tensors, layers),
            lambda tensors: _concat_compressor_gates(tensors, layers),
            lambda tensors: interleave_gate_up(tensors, layers, prefix="ffn"),
            lambda tensors: concat_gate_up(tensors, layers, prefix="ffn.shared_experts"),
        ],
        dtype,
    )


def _group_output_lora(
    weights: dict[str, mx.array], layers: int, groups: int, rank: int
) -> dict[str, mx.array]:
    """`wo_a` is block-diagonal in 8 groups, serialized as one 2D matrix. Reshaped into the
    `[groups, out, in]` a `SwitchLinear` declares, it is the same numbers in the same order
    — a block-diagonal matmul *is* a gather with every slot active."""
    for layer in range(layers):
        for suffix in ("weight", "scales", "biases"):
            key = f"model.layers.{layer}.attn.wo_a.{suffix}"
            tensor = weights.get(key)
            if tensor is not None and tensor.ndim == 2:
                weights[key] = tensor.reshape(groups, rank, -1)
    return weights


def _concat_compressor_gates(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`wkv` and `wgate` rows stacked into one `wkvg` per compressor: a decode step pays
    one gemv instead of two, and a row concat leaves every quantization group intact."""
    for layer in range(layers):
        for prefix in ("attn.compressor", "attn.indexer.compressor"):
            base = f"model.layers.{layer}.{prefix}"
            for suffix in ("weight", "scales", "biases"):
                kv = weights.pop(f"{base}.wkv.{suffix}", None)
                gate = weights.pop(f"{base}.wgate.{suffix}", None)
                if kv is not None and gate is not None:
                    weights[f"{base}.wkvg.{suffix}"] = mx.concatenate([kv, gate], axis=0)
    return weights


def _narrow_hash_tables(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`tid2eid` ships as int64: one expert index per token id, and the vocabulary is
    129280 rows. int32 indexes the same 256 experts at half the bytes."""
    for layer in range(layers):
        key = f"model.layers.{layer}.ffn.gate.tid2eid"
        if key in weights:
            weights[key] = weights[key].astype(mx.int32)
    return weights


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {key: value for key, value in weights.items() if not key.startswith("mtp.")}


def _composite(directory: Path, model: DeepseekV4) -> LanguageModel[ModelInput]:
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
    DeepseekV4Config,
    DeepseekV4,
    weights,
    _composite,
    model_types=("deepseek_v4",),
)
