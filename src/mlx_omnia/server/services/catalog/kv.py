"""What one token costs the key-value cache, and where that cache stops growing."""

from __future__ import annotations

from pathlib import Path

from mlx_omnia.server.services.catalog.config import ConfigJson, ShapeJson, shape_of
from mlx_omnia.server.services.catalog.headers import weights_dtype

_ATTENDING = frozenset({"full_attention", "sliding_attention", "attention"})
"""What transformers calls a layer that keeps a key-value cache. Everything else a
`layer_types` names holds a state of fixed size, so it is not in this arithmetic."""

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}


def _attending_layers(shape: ShapeJson) -> int | None:
    """How many layers keep a growing cache, in whichever of the three forms the config says
    it: every layer named, the attending ones listed, or the stride of a repeating group.

    With a stride, one layer of each group attends — the one that closes it — and a trailing
    group the stride does not fill is attention throughout.
    """
    layers = shape.get("num_hidden_layers")
    if layers is None:
        return None
    if (types := shape.get("layer_types")) is not None:
        return sum(1 for kind in types if kind in _ATTENDING)
    if (indices := shape.get("full_attn_idxs")) is not None:
        return len(indices)
    if (group := shape.get("layer_group_size")) is not None and group > 0:
        return layers // group + layers % group
    return layers


def head_width(shape: ShapeJson) -> int | None:
    """How wide one row of one attending layer's cache is: a latent cache's compressed vector
    plus the rotated key, or one ordinary head."""
    if (rank := shape.get("kv_lora_rank")) is not None:
        rope = shape.get("qk_rope_head_dim")
        return None if rope is None else rank + rope
    if (head_dim := shape.get("head_dim")) is not None:
        return head_dim
    heads = shape.get("num_attention_heads")
    hidden = shape.get("hidden_size")
    if heads is None or heads == 0 or hidden is None:
        return None
    return hidden // heads


def _elements_per_layer(shape: ShapeJson) -> int | None:
    """Elements one token adds to one attending layer's cache: one row for a latent cache, a
    key plus a value per key-value head for the ordinary one."""
    width = head_width(shape)
    if width is None:
        return None
    if shape.get("kv_lora_rank") is not None:
        return width
    heads = shape.get("num_attention_heads")
    if heads is None:
        return None
    return 2 * shape.get("num_key_value_heads", heads) * width


def bytes_per_token(directory: Path, config: ConfigJson) -> int | None:
    """What one token costs the cache, over the whole model. The element size comes off the
    shards: a 4-bit checkpoint computes and caches in the bfloat16 its norms are stored in."""
    shape = shape_of(config)
    layers = _attending_layers(shape)
    elements = _elements_per_layer(shape)
    if layers is None or elements is None:
        return None
    width = _DTYPE_BYTES.get(weights_dtype(directory) or "")
    return None if width is None else layers * elements * width


def attention_window(shape: ShapeJson) -> int | None:
    """The context past which the cache stops growing, and `None` whenever it does not stop.

    Only when every attending layer slides. A checkpoint that alternates — gpt-oss keeps a
    full layer beside each sliding one — has a cache that still grows linearly.
    """
    window = shape.get("sliding_window")
    if window is None or not shape.get("use_sliding_window", True):
        return None
    types = shape.get("layer_types")
    if types is None:
        return window
    attending = [kind for kind in types if kind in _ATTENDING]
    return window if attending and all(kind == "sliding_attention" for kind in attending) else None
