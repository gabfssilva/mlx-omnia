from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    checkpoint,
    drop_tied_head,
    interleave_gate_up,
    load_shards,
    materialize,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.longcat_flash_ngram.config import LongcatFlashNgramConfig
from mlx_omnia.engine.models.longcat_flash_ngram.model import LongcatFlashNgram


def weights(
    directory: Path, config: LongcatFlashNgramConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    loaded = load_shards(directory)
    reject_dtype_cast(dtype, loaded)
    if dtype is not None:
        loaded = {
            key: value if "e_score_correction_bias" in key else value.astype(dtype)
            for key, value in loaded.items()
        }
    if config.tie_word_embeddings:
        drop_tied_head(loaded)
    loaded = _stack_experts(loaded, config)
    loaded = _concat_dense_gate_up(loaded, config)
    loaded = _split_kv_b(loaded, config)
    loaded = _drop_mtp(loaded)
    loaded = _rename_embed(loaded)
    loaded = interleave_gate_up(loaded, config.num_layers)
    return loaded


def _stack_experts(
    weights: dict[str, mx.array], config: LongcatFlashNgramConfig
) -> dict[str, mx.array]:
    """Stack per-expert ``mlp.experts.{e}.{gate,up,down}_proj.*`` into
    ``mlp.switch_mlp.{gate,up,down}_proj.*`` (only the routed experts;
    identity experts carry no weights)."""
    for layer in range(config.num_layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}0.{proj}.{suffix}"
                if key not in weights:
                    continue
                stacked = mx.stack(
                    [
                        weights.pop(f"{prefix}{e}.{proj}.{suffix}")
                        for e in range(config.n_routed_experts)
                    ]
                )
                materialize(stacked)
                weights[f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"] = stacked
    return weights


def _concat_dense_gate_up(
    weights: dict[str, mx.array], config: LongcatFlashNgramConfig
) -> dict[str, mx.array]:
    """Concatenate gate‖up for the dense MLPs (``mlps.{i}.``), the sibling of
    ``concat_gate_up`` for the ``mlps.{i}.`` path the spine does not know."""
    for layer in range(config.num_layers):
        for i in range(2):
            prefix = f"model.layers.{layer}.mlps.{i}."
            for suffix in ("weight", "scales", "biases"):
                keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
                if not all(key in weights for key in keys):
                    continue
                fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
                materialize(fused)
                weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _split_kv_b(
    weights: dict[str, mx.array], config: LongcatFlashNgramConfig
) -> dict[str, mx.array]:
    """Split each sublayer's ``kv_b_proj`` into ``embed_q`` + ``unembed_out``.

    If the source is quantized, dequantize before the split and leave the halves
    dense — the shared ``_quantization`` predicate does not recognize
    ``MultiLinear``, so re-quantizing would orphan the ``.scales``/``.biases``.
    """
    heads = config.num_attention_heads
    nope = config.qk_nope_head_dim
    v_head = config.v_head_dim
    kv_lora = config.kv_lora_rank

    for layer in range(config.num_layers):
        for i in range(2):
            prefix = f"model.layers.{layer}.self_attn.{i}"
            kv_b_key = f"{prefix}.kv_b_proj.weight"
            if kv_b_key not in weights:
                continue
            quantized = f"{prefix}.kv_b_proj.scales" in weights
            v = weights.pop(kv_b_key)
            if quantized:
                scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                bits = (v.shape[-1] * 32) // kv_lora
                group_size = kv_lora // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

            v = v.reshape(heads, nope + v_head, kv_lora)
            wk = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, nope:, :])
            materialize(wk, wv)
            weights[f"{prefix}.embed_q.weight"] = wk
            weights[f"{prefix}.unembed_out.weight"] = wv
    return weights


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}


def _rename_embed(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    key = "model.embed_tokens.weight"
    if key in weights:
        weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(key)
    return weights


def _composite(directory: Path, model: LongcatFlashNgram) -> LanguageModel[ModelInput]:
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
    LongcatFlashNgramConfig,
    LongcatFlashNgram,
    weights,
    _composite,
    model_types=("longcat_flash_ngram",),
)
