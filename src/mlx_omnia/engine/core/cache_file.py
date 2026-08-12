"""A trunk's cache as one safetensors file, and back into a trunk that has just been made.

Next to `PromptCache` and not inside `LayerCache` because none of the three things that make
this hard is a property of one layer. The list is heterogeneous — a hybrid holds attention
beside recurrence — so what walks it is a function over `Sequence[LayerCache]`, and what a
layer answers for is its own tensors and nothing else.

What is written is `[..., :offset, :]`: the block-grown buffer reserves up to 255 rows past
the offset, and rows a model never wrote are not state. What is read back is exactly as long
as it was written, which `core.cache._reserving` grows on the next update like any buffer too
small for what is coming.

The offsets ride in the file's string metadata rather than as tensors of one element: they
are the one part of the state a reader has to trust before it has read anything, and a
`__meta__` block is what safetensors gives for that.

Reading ends with `mx.eval`, for trap 2 of the house: `mx.load` is lazy over an mmap, and the
first evaluation of a lazily mapped tensor comes out corrupted.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.core.cache import LayerCache

_OFFSETS = "offsets"
"""Where the per-layer row counts ride. One key holding a JSON list rather than one key per
layer: what has to be checked on the way in is that the file describes *this* trunk, and a
list has a length."""


class UnreadableCache(Exception):
    """The file is not a cache this trunk can take: a different number of layers, a layer
    that answered with tensors this one does not know, or bytes that are not a cache at all.

    Named rather than swallowed, and the caller's business what to do with it — a prefix
    store treats it as a miss and prefills, which is exactly right and is a decision about
    the cache, not about the file."""


def storable(caches: Sequence[LayerCache]) -> bool:
    """Whether this trunk's cache can go to a file at all. Over the whole list, the way
    `is_trimmable` is read: a compiled or ring cache does not answer, and a list written
    short of one layer restores a trunk whose layers disagree about how much they have
    seen."""
    return all(layer.is_storable for layer in caches)


def policy(caches: Sequence[LayerCache]) -> dict[str, object]:
    """What this trunk's bytes mean, for `key` to hash beside the ids.

    Collected from the layers rather than from a caller's configuration, because the layer is
    the only thing that knows: a compressed cache carries its codec, a ring carries its
    rotation, and a growing one carries nothing because its rows are what the model produced.
    Per layer and not per trunk — a hybrid compresses attention and not recurrence, and one
    number for the whole list would call two policies the same.

    A trunk of layers that all answer nothing produces `{}`, which is the value that was in
    the key before any layer had anything to say. That is deliberate: it keeps every file
    already written under a dense trunk hitting instead of being orphaned by a new field.
    """
    signatures = [sorted(layer.signature.items()) for layer in caches]
    return {"layers": signatures} if any(signatures) else {}


def dump(caches: Sequence[LayerCache], path: Path) -> None:
    """Write the list to `path`, atomically: a staging file in the same directory and a
    rename, so a reader never opens a half-written cache. The same shape `download._install`
    and the quantizer use, and for the same reason — a truncated file that is never visible
    beats one that is visible and has to be detected."""
    tensors: dict[str, mx.array] = {}
    for index, layer in enumerate(caches):
        for name, tensor in layer.stored().items():
            tensors[f"{index}.{name}"] = tensor
    offsets = [layer.rows for layer in caches]
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique staging name in the same directory, and a suffix mlx will not append to: two
    # daemons over one cache directory write the same key at the same time, and a shared
    # staging name would let one rename the other's half-written file into place.
    handle, staging = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(handle)
    mx.save_safetensors(staging, tensors, metadata={_OFFSETS: json.dumps(offsets)})
    Path(staging).replace(path)


def restore(caches: Sequence[LayerCache], path: Path) -> None:
    """Fill `caches` — a list the model has just made — from `path`.

    Raises `UnreadableCache` and leaves the list untouched when the file does not describe
    this trunk. Untouched matters: a caller that catches the error and prefills instead must
    not be prefilling on top of half a restore.
    """
    try:
        tensors, metadata = mx.load(str(path), return_metadata=True)
        offsets = json.loads(metadata[_OFFSETS])
    except Exception as unreadable:
        raise UnreadableCache(f"{path.name} is not a readable cache file") from unreadable
    if not isinstance(tensors, dict):
        raise UnreadableCache(f"{path.name} holds no tensors")
    if not isinstance(offsets, list) or len(offsets) != len(caches):
        raise UnreadableCache(
            f"{path.name} describes {len(offsets) if isinstance(offsets, list) else '?'} "
            f"layers and this trunk has {len(caches)}"
        )
    grouped: list[dict[str, mx.array]] = [{} for _ in caches]
    for key, tensor in tensors.items():
        index, _, name = key.partition(".")
        if not index.isdigit() or int(index) >= len(caches) or not name:
            raise UnreadableCache(f"{path.name} names a layer this trunk does not have: {key}")
        grouped[int(index)][name] = tensor
    for layer, offset, held in zip(caches, offsets, grouped, strict=True):
        if not isinstance(offset, int):
            raise UnreadableCache(f"{path.name} carries an offset that is not a number")
        layer.restore(offset, held)
    mx.eval([tensor for layer in caches for tensor in layer.tensors])


def written(caches: Sequence[LayerCache]) -> int:
    """What `dump` would put on disk, before writing it. What a disk ceiling is enforced
    against, and it is not `nbytes`: that one counts the reserved buffers, and this counts
    the rows in use."""
    return sum(tensor.nbytes for layer in caches for tensor in layer.stored().values())


def read_offsets(path: Path) -> Sequence[int] | None:
    """The row counts a file carries, without reading a tensor. What an index rebuilt off a
    directory reads, and what a caller checks a candidate's length against before paying for
    the mmap."""
    try:
        _, metadata = mx.load(str(path), return_metadata=True)
        offsets = json.loads(metadata[_OFFSETS])
    except Exception:
        return None
    return offsets if isinstance(offsets, list) else None


def key(model_id: str, stamp: str, policy: Mapping[str, object], tokens: Sequence[int]) -> str:
    """What separates a hit from a disaster. Four things, because each of them changes what
    the bytes mean and none of them is visible in the file:

    - the model id, since the ids of two checkpoints are two alphabets;
    - the checkpoint's stamp, which is what moves when a shard is replaced under the same id;
    - the KV policy, because a dense cache and a compressed one are two formats of the same
      thing and only one of them is what the trunk about to read it builds;
    - the exact ids, which is the prefix itself.

    A digest and not a path made of them: the ids are the long part, and a file name is not
    where a conversation should be legible on disk.
    """
    material = json.dumps(
        [model_id, stamp, sorted(policy.items()), list(tokens)], separators=(",", ":")
    )
    return hashlib.sha256(material.encode()).hexdigest()
