"""Safetensors header arithmetic: what the shards carry, and what one decode step reads."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from mlx_omnia.engine.footprint import expert_slots
from mlx_omnia.engine.task import source as checkpoint_source
from mlx_omnia.server.services.catalog.config import (
    NO_TEXT,
    ConfigJson,
    IndexJson,
    TensorJson,
)

_EXPERT_SLICE = re.compile(r"(?:^|\.)experts\.\d+\.")
"""A conversion shipping one tensor per expert instead of the stack (LFM2, Laguna). Either
way a step reads `k` of the `E` rows."""

_FLOATS = ("BF16", "F16", "F32", "F64")


def tensors_of(directory: Path) -> list[tuple[str, TensorJson, int]] | None:
    """Every tensor's name, header entry and physical bytes, with no shard mapped. `None` when
    a header does not parse.

    The declared length is checked against the file before it is read: on arbitrary bytes
    those first eight are an arbitrary u64, and reading it is a `MemoryError`.
    """
    found: list[tuple[str, TensorJson, int]] = []
    for shard in sorted(directory.glob("model*.safetensors")):
        try:
            with shard.open("rb") as file:
                length = int.from_bytes(file.read(8), "little")
                if length > shard.stat().st_size - 8:
                    return None
                header: dict[str, TensorJson] = json.loads(file.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                begin, end = entry["data_offsets"]
                found.append((name, entry, end - begin))
        except (ValueError, KeyError):
            return None
    return found


def stored_carrier(directory: Path) -> str | None:
    """The dtype carrying most of the shards' bytes. For one quantized natively — I8 or U32
    codes beside a sliver of float scales, with nothing in the config saying so — it is the
    codes' dtype, which is how a caller tells a checkpoint that only looks dense from one
    that is."""
    tensors = tensors_of(directory)
    if not tensors:
        return None
    weighed: dict[str, int] = {}
    for _, entry, size in tensors:
        weighed[entry["dtype"]] = weighed.get(entry["dtype"], 0) + size
    return max(weighed.items(), key=lambda pair: pair[1])[0]


def weights_dtype(directory: Path) -> str | None:
    """The dtype the shards' floating tensors carry, under safetensors' own name. The config
    is not a reliable source: qwen3.5 nests `dtype` under `text_config` and some conversions
    declare none. The largest tensor decides."""
    tensors = tensors_of(directory)
    if not tensors:
        return None
    floating = [(size, entry["dtype"]) for _, entry, size in tensors if entry["dtype"] in _FLOATS]
    return max(floating)[1] if floating else None


@lru_cache(maxsize=256)
def _slots(directory: Path, stamp: int) -> Mapping[int, int] | None:
    """How many rows of an expert stack a step reads, by the stack's depth, off the tree the
    architecture builds from its own config.

    `None` when there is no such tree, and then a checkpoint that stacks anything is not
    priced at all: reading a key and defaulting to zero charged every expert on every token,
    and reported a decode rate against a ceiling an order of magnitude too low.
    """
    del stamp
    try:
        return expert_slots(checkpoint_source(directory, local_files_only=True).pending.model)
    except Exception:
        return None


def bytes_per_token(directory: Path, config: ConfigJson) -> int | None:
    """What one decode step reads, summed from the headers: `k` of the `E` rows of a stack and
    never the whole pile, the untied lm_head whole, the embedding table only when it is also
    the head, and never the vision tower's position table.

    Something stacked that no tree answers for is not priced at all — that is the difference
    between an estimate and an invented number.
    """
    text = config.get("text_config", NO_TEXT)
    vocab = text.get("vocab_size") or config.get("vocab_size") or 0
    nested = text.get("tie_word_embeddings")
    tied = config.get("tie_word_embeddings", True) if nested is None else nested
    tensors = tensors_of(directory)
    if tensors is None:
        return None
    read = 0
    stacked: list[tuple[int, int]] = []
    sliced: dict[str, list[int]] = {}
    for name, entry, size in tensors:
        shape = entry["shape"]
        # gpt2's causal-mask buffer, which its loader drops before the tree ever sees it.
        if name.endswith(".attn.bias"):
            continue
        head = name.split(".")[-2:-1] == ["lm_head"]
        if tied and head:
            continue
        if name.split(".")[-2:-1] == ["pos_embed"]:
            continue
        if not tied and not head and len(shape) >= 2 and shape[0] == vocab:
            continue
        if _EXPERT_SLICE.search(name):
            sliced.setdefault(_EXPERT_SLICE.sub(".experts.", name), []).append(size)
        elif len(shape) >= 3:
            stacked.append((shape[0], size))
        else:
            read += size
    if not stacked and not sliced:
        return read
    slots = _slots(directory, (directory / "config.json").stat().st_mtime_ns)
    if slots is None:
        return None
    for rows, size in stacked:
        # A depth no stack has is not a stack: a conv1d's weight is three-dimensional too,
        # and a step reads it whole.
        read += size * slots[rows] // rows if rows in slots else size
    for group in sliced.values():
        rows = len(group)
        if rows not in slots:
            return None
        read += sum(group) * slots[rows] // rows
    return read


def complete(directory: Path) -> bool:
    """The `weight_map` is the list the loader will ask for, so it is the list that decides.
    `is_file` follows the link, which catches a snapshot pointing at a blob the download never
    finished."""
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return (directory / "model.safetensors").is_file()
    weights: IndexJson = json.loads(index.read_text())
    return all((directory / shard).is_file() for shard in set(weights["weight_map"].values()))
