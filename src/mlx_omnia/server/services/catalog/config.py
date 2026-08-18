"""The shapes a checkpoint's `config.json` is read through, and where its numbers are."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from mlx_omnia.engine.checkpoint import QuantizationJson


class _QuantizationJson(TypedDict):
    """Both shapes the block is written in: mlx's global one and the per-leaf plan
    `save_quantized` writes."""

    bits: NotRequired[int]
    mode: NotRequired[str]
    leaves: NotRequired[dict[str, QuantizationJson]]


class ShapeJson(TypedDict):
    """The keys the arithmetic reads, all of them transformers' own names. They appear at the
    root or under `text_config`."""

    max_position_embeddings: NotRequired[int]
    vocab_size: NotRequired[int]
    tie_word_embeddings: NotRequired[bool]
    hidden_size: NotRequired[int]
    num_hidden_layers: NotRequired[int]
    num_attention_heads: NotRequired[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    sliding_window: NotRequired[int | None]
    use_sliding_window: NotRequired[bool]
    layer_types: NotRequired[list[str]]
    full_attn_idxs: NotRequired[list[int]]
    layer_group_size: NotRequired[int]
    kv_lora_rank: NotRequired[int]
    qk_rope_head_dim: NotRequired[int]


class TextConfigJson(ShapeJson):
    pass


class ConfigJson(ShapeJson):
    model_type: NotRequired[str]
    dtype: NotRequired[str]
    torch_dtype: NotRequired[str]
    quantization: NotRequired[_QuantizationJson]
    text_config: NotRequired[TextConfigJson]


class TensorJson(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class IndexJson(TypedDict):
    weight_map: dict[str, str]


NO_TEXT: TextConfigJson = {}


def shape_of(config: ConfigJson) -> ShapeJson:
    """Where the architecture's numbers are, root or nested. A key present at both is the
    nested one's."""
    text = config.get("text_config")
    if text is None:
        return config
    merged: ShapeJson = {**config, **text}
    return merged


def _label(bits: int, mode: str) -> str:
    return f"{bits}-bit" if mode == "affine" else mode


def quantization_of(config: ConfigJson) -> str | None:
    """What the config says it is, never what the tensors are. A per-leaf plan reports the
    width its leaves agree on, and `mixed` when they don't."""
    block = config.get("quantization")
    if block is None:
        return None
    if (leaves := block.get("leaves")) is not None:
        labels = {_label(leaf["bits"], leaf.get("mode", "affine")) for leaf in leaves.values()}
        return labels.pop() if len(labels) == 1 else "mixed"
    bits = block.get("bits")
    return None if bits is None else _label(bits, block.get("mode", "affine"))


def print_of(config: ConfigJson) -> str | None:
    """`model_type` and the four numbers that decide whether two checkpoints are the same
    architecture. Not a fingerprint of the weights: a fine-tune prints the same as its
    base."""
    shape = shape_of(config)
    architecture = config.get("model_type")
    layers = shape.get("num_hidden_layers")
    hidden = shape.get("hidden_size")
    heads = shape.get("num_attention_heads")
    vocab = shape.get("vocab_size")
    if architecture is None or None in (layers, hidden, heads, vocab):
        return None
    return f"{architecture}/L{layers}/H{hidden}/A{heads}/V{vocab}"
