"""One span's tensors as one safetensors file, and back.

A payload and not a trunk: what is stored is a span — an immutable slice of one
conversation, named `"{layer}.{tensor}"` — so nothing here knows how many layers a model has
or what any of them holds. The composition back into a trunk is `core.prefix`'s, over the
layouts the layers declare, and this is only the bytes.

The ids of the span ride in the payload as a tensor. They are ~1 KB against the megabytes
beside them and they are what makes a digest collision a miss instead of a disaster: a reader
compares them against the ids it asked for, and a file that answers with somebody else's
tokens is forgotten.

Writing is atomic — a staging file in the same directory and a rename — so a reader never
opens a half-written span, and two daemons writing the same content-addressed key write the
same bytes and replace each other harmlessly.

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

EMPTY = "__empty__"
"""Where the tensors with no elements ride: their shape and dtype, as JSON.

safetensors cannot serialize an array of zero elements, and a cache legitimately holds some —
`deepseek_v4` reads K and V out of one buffer, so its `values` is a placeholder of width 0.
Dropping them and rebuilding them on the way in keeps the layer's contract whole: what comes
back is named what it was named and shaped how it was shaped, and the workaround stays inside
the one thing that has the limitation."""

IDS = "__ids__"
"""Where the span's own tokens ride. Inside the payload rather than beside it because the
check it exists for happens after the read: whoever verifies has the bytes in hand."""


class UnreadableCache(Exception):
    """The file is not a payload: bytes that are not safetensors, or a payload with no ids in
    it. Named rather than swallowed, and the caller's business what to do with it — a prefix
    store treats it as a miss and prefills, which is exactly right and is a decision about
    the cache, not about the file."""


def policy(caches: Sequence[LayerCache]) -> dict[str, object]:
    """What this trunk's bytes mean, for the chain's seed to hash.

    Collected from the layers rather than from a caller's configuration, because the layer is
    the only thing that knows: a compressed cache carries its codec, a pooled one its ratio.
    Per layer and not per trunk — a hybrid compresses attention and not recurrence, and one
    number for the whole list would call two policies the same.
    """
    signatures = [sorted(layer.signature.items()) for layer in caches]
    return {"layers": signatures} if any(signatures) else {}


def dump(payload: Mapping[str, mx.array], path: Path) -> None:
    """Write one payload to `path`, atomically."""
    held = {name: tensor for name, tensor in payload.items() if tensor.size}
    empty = {
        name: [list(tensor.shape), _spelled(tensor.dtype)]
        for name, tensor in payload.items()
        if not tensor.size
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique staging name in the same directory, and a suffix mlx will not append to: two
    # daemons over one cache directory write the same key at the same time, and a shared
    # staging name would let one rename the other's half-written file into place.
    handle, staging = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(handle)
    mx.save_safetensors(staging, held, metadata={EMPTY: json.dumps(empty)})
    Path(staging).replace(path)


def load(path: Path) -> dict[str, mx.array]:
    """Read one payload back, evaluated. Raises `UnreadableCache` for anything that is not
    one, which a caller turns into a miss."""
    try:
        tensors, metadata = mx.load(str(path), return_metadata=True)
        empty = json.loads(metadata.get(EMPTY, "{}"))
    except Exception as unreadable:
        raise UnreadableCache(f"{path.name} is not a readable payload") from unreadable
    if not isinstance(tensors, dict) or IDS not in tensors:
        raise UnreadableCache(f"{path.name} carries no span ids")
    if not isinstance(empty, dict):
        raise UnreadableCache(f"{path.name} describes its empty tensors as something else")
    for name, described in empty.items():
        shape, dtype = described
        tensors[name] = mx.zeros(shape, dtype=_DTYPES[dtype])
    mx.eval(list(tensors.values()))
    return tensors


def weight(payload: Mapping[str, mx.array]) -> int:
    """What a payload costs whichever ceiling holds it. The same number in memory and on
    disk, because a span is the rows themselves and safetensors adds a header and no
    padding."""
    return sum(tensor.nbytes for tensor in payload.values())


def _spelled(dtype: mx.Dtype) -> str:
    """A dtype as one name. `str(mx.float32)` is `mlx.core.float32`, and what a file carries
    has to be the same string next release."""
    return str(dtype).rpartition(".")[2]


_DTYPES: Mapping[str, mx.Dtype] = {
    _name: getattr(mx, _name)
    for _name in (
        "bool_",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
    )
}


def digest(material: object) -> str:
    """The house's hash over anything JSON can spell, for the chain to link with.

    A digest and not a path made of the material: the ids are the long part, and a file name
    is not where a conversation should be legible on disk."""
    return hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
