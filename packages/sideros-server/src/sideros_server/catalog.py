"""The models on disk: the Hugging Face cache and what `quantize=` leaves behind.

An entry's id is what `sideros.load` takes back. A hub repository is its repository id,
listed once at the revision `refs/main` names; a quantized entry has no repository name of
its own — `task.py` derives its folder from a fingerprint of the source — so its id is the
directory itself, which is the other thing `load` accepts.

What decides that a directory is a model is the `weight_map` of
`model.safetensors.index.json`. An interrupted download keeps its config and some of its
shards, and listing it turns a failed download into a load error at request time.

`bytes_per_token` is priced per entry, not cached: the arithmetic reads the safetensors
headers and never a shard, which on this machine's cache is 0.9 ms per entry warm (5.4 ms
cold on a 4-shard 17 GB checkpoint) against the 25 ms the walk already costs for 74
entries. The lazy tree would have been the other source, and it cannot be: it is built
before `nn.quantize`, so a 4-bit checkpoint would be priced at its dense fp32 shapes.

The handlers are sync on purpose: the scan stats a few hundred files and a delete can
unlink tens of gigabytes, so they run in the threadpool instead of stalling the loop —
and with it the token stream of whatever is generating.
"""

import json
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Annotated, NotRequired, TypedDict

import huggingface_hub.constants
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from sideros.checkpoint import QuantizationJson
from sideros_server.engine import Engine

HUB_CACHE = Path(huggingface_hub.constants.HF_HUB_CACHE)
QUANTIZED_CACHE = Path.home() / ".cache" / "sideros" / "quantized"


class _QuantizationJson(TypedDict):
    """Both shapes the block is written in: mlx's global one and the per-leaf plan
    `save_quantized` writes. Neither key is required — a dense config has no block at all,
    and mlx-lm's per-path overrides sit next to the global keys under names we do not
    read."""

    bits: NotRequired[int]
    mode: NotRequired[str]
    leaves: NotRequired[dict[str, QuantizationJson]]


class _TextConfigJson(TypedDict):
    """Where qwen3.5 keeps everything a text checkpoint declares, including the fields the
    per-token arithmetic reads."""

    max_position_embeddings: NotRequired[int]
    vocab_size: NotRequired[int]
    tie_word_embeddings: NotRequired[bool]
    num_experts: NotRequired[int]
    num_local_experts: NotRequired[int]
    num_experts_per_tok: NotRequired[int]


class _ConfigJson(TypedDict):
    model_type: NotRequired[str]
    max_position_embeddings: NotRequired[int]
    dtype: NotRequired[str]
    torch_dtype: NotRequired[str]
    vocab_size: NotRequired[int]
    tie_word_embeddings: NotRequired[bool]
    num_experts: NotRequired[int]
    num_local_experts: NotRequired[int]
    num_experts_per_tok: NotRequired[int]
    quantization: NotRequired[_QuantizationJson]
    text_config: NotRequired[_TextConfigJson]


class _TensorJson(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class _IndexJson(TypedDict):
    weight_map: dict[str, str]


_NO_TEXT: _TextConfigJson = {}

_EXPERT_SLICE = re.compile(r"(?:^|\.)experts\.\d+\.")
"""A conversion that ships one tensor per expert instead of the stack (LFM2, Laguna). The
loader stacks them into `[E, out, in]`; either way a step reads `k` of the `E` rows."""

_FLOATS = ("BF16", "F16", "F32", "F64")


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    directory: Path
    """The checkpoint: `config.json` and the shards."""
    store: Path
    """What the id owns on disk, and what DELETE removes. A hub repository owns its whole
    folder: the snapshot is symlinks into `blobs/`, and unlinking it alone frees nothing."""
    architecture: str
    quantization: str | None
    """The label the config declares — `None` when it declares none, which is dense."""
    dtype: str | None
    context: int | None
    bytes_on_disk: int
    bytes_per_token: int | None = None
    """Bytes a decode step reads, and the denominator of every "% of the ceiling" the house
    reports. `None` when the shards' headers cannot be read."""
    resident: bool = False
    """Filled by the handler from the engine, not by the scan."""


def _label(bits: int, mode: str) -> str:
    return f"{bits}-bit" if mode == "affine" else mode


def _quantization(config: _ConfigJson) -> str | None:
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


def _tensors(directory: Path) -> list[tuple[str, _TensorJson, int]] | None:
    """Every tensor's name, header entry and physical bytes, read off the safetensors
    headers with no shard mapped: each file opens with a little-endian u64 holding the
    header's length, and `__metadata__` sits among the tensors carrying no offsets.

    `None` when a header does not parse. The scan lists what is on disk, and a file that is
    not safetensors costs the entry its numbers, not its listing. The length is checked
    against the file before it is read: on arbitrary bytes those first eight are an
    arbitrary u64, and reading it is a `MemoryError` rather than a bad JSON.
    """
    found: list[tuple[str, _TensorJson, int]] = []
    for shard in sorted(directory.glob("model*.safetensors")):
        try:
            with shard.open("rb") as file:
                length = int.from_bytes(file.read(8), "little")
                if length > shard.stat().st_size - 8:
                    return None
                header: dict[str, _TensorJson] = json.loads(file.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                begin, end = entry["data_offsets"]
                found.append((name, entry, end - begin))
        except (ValueError, KeyError):
            return None
    return found


def weights_dtype(directory: Path) -> str | None:
    """The dtype the shards' floating tensors carry, under the name safetensors gives it
    (`BF16`, `F16`, `F32`). It is what a quantization plan has to be priced in, and the
    config is not a reliable source for it: qwen3.5 nests `dtype` under `text_config` and
    some conversions declare none at all. The largest tensor decides — a checkpoint mixing
    float widths across its weights does not exist, and the odd float32 norm beside
    bfloat16 matrices must not."""
    tensors = _tensors(directory)
    if not tensors:
        return None
    floating = [(size, entry["dtype"]) for _, entry, size in tensors if entry["dtype"] in _FLOATS]
    return max(floating)[1] if floating else None


def _bytes_per_token(directory: Path, config: _ConfigJson) -> int | None:
    """What one decode step reads, summed from the headers: `k` of the `E` experts and never
    the whole stack, the untied lm_head whole, the embedding table only when it is also the
    head — a step gathers one row of it, which is free — and never the vision tower's
    position table, which a text step does not reach.

    Three facts the tensors do not carry come off the config. How many experts a token
    reaches. Whether the head is tied, which transformers defaults to true and gpt2 leaves
    unsaid. And the vocabulary, which is what tells the embedding table apart from the head:
    both are `[vocab, ·]`, and only one of them is named `lm_head`.
    """
    text = config.get("text_config", _NO_TEXT)
    vocab = text.get("vocab_size") or config.get("vocab_size") or 0
    nested = text.get("tie_word_embeddings")
    tied = config.get("tie_word_embeddings", True) if nested is None else nested
    reached = text.get("num_experts_per_tok") or config.get("num_experts_per_tok") or 0
    experts = (
        text.get("num_experts")
        or config.get("num_experts")
        or text.get("num_local_experts")
        or config.get("num_local_experts")
        or 0
    )
    tensors = _tensors(directory)
    if tensors is None:
        return None
    read = 0
    routed = 0
    for name, entry, size in tensors:
        shape = entry["shape"]
        # gpt2's causal-mask buffer, which its loader drops before the tree ever sees it.
        if name.endswith(".attn.bias"):
            continue
        head = name.split(".")[-2:-1] == ["lm_head"]
        if tied and head:
            continue
        # The tables a step gathers rows of instead of reading: the tokens', when it is not
        # also the head, and the vision tower's positions — which a text step does not
        # reach at all, and which the tree holds as the `nn.Embedding` it is.
        if name.split(".")[-2:-1] == ["pos_embed"]:
            continue
        if not tied and not head and len(shape) >= 2 and shape[0] == vocab:
            continue
        stacked = len(shape) >= 3 and shape[0] == experts
        if experts and (stacked or _EXPERT_SLICE.search(name)):
            routed += size
        else:
            read += size
    return read + (routed * reached // experts if experts else 0)


def _complete(directory: Path) -> bool:
    """The `weight_map` is the list the loader will ask for, so it is the list that
    decides. `is_file` follows the link, which is what catches a snapshot pointing at a
    blob the download never finished."""
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return (directory / "model.safetensors").is_file()
    weights: _IndexJson = json.loads(index.read_text())
    return all((directory / shard).is_file() for shard in set(weights["weight_map"].values()))


def _entry(model_id: str, directory: Path, store: Path) -> CatalogEntry | None:
    config_path = directory / "config.json"
    if not config_path.is_file() or not _complete(directory):
        return None
    config: _ConfigJson = json.loads(config_path.read_text())
    architecture = config.get("model_type")
    if architecture is None:
        return None
    context = config.get("max_position_embeddings")
    if context is None and (text := config.get("text_config")) is not None:
        context = text.get("max_position_embeddings")
    return CatalogEntry(
        id=model_id,
        directory=directory,
        store=store,
        architecture=architecture,
        quantization=_quantization(config),
        dtype=config.get("dtype") or config.get("torch_dtype"),
        context=context,
        bytes_on_disk=sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
        bytes_per_token=_bytes_per_token(directory, config),
    )


def _head(repository: Path) -> Path | None:
    """The revision `load` would resolve. A snapshot fetched by sha has no `refs/main`, so
    the fallback is the most recent one rather than nothing."""
    snapshots = repository / "snapshots"
    if not snapshots.is_dir():
        return None
    reference = repository / "refs" / "main"
    sha = reference.read_text().strip() if reference.is_file() else ""
    if sha and (head := snapshots / sha).is_dir():
        return head
    revisions = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    return revisions[-1] if revisions else None


def _hub(root: Path) -> Iterator[CatalogEntry]:
    if not root.is_dir():
        return
    for repository in root.glob("models--*"):
        head = _head(repository)
        if head is None:
            continue
        model_id = repository.name.removeprefix("models--").replace("--", "/")
        if (entry := _entry(model_id, head, repository)) is not None:
            yield entry


def _quantized(root: Path) -> Iterator[CatalogEntry]:
    if not root.is_dir():
        return
    for source in root.iterdir():
        if not source.is_dir():
            continue
        for directory in source.iterdir():
            # `write_entry` stages under `.tmp-*` next to the entry it is about to become.
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            if (entry := _entry(str(directory), directory, directory)) is not None:
                yield entry


def scan() -> list[CatalogEntry]:
    return sorted([*_hub(HUB_CACHE), *_quantized(QUANTIZED_CACHE)], key=lambda entry: entry.id)


@lru_cache(maxsize=256)
def context_of(model_id: str) -> int | None:
    """The `max_position_embeddings` the checkpoint declares, by id — what a dialect caps a
    generation with. `None` for an id the disk does not answer for, which is every test
    double. Cached because it rides the request path and a checkpoint's context does not
    move; the scan behind it prices every entry, and the cache is what keeps that off every
    chat turn."""
    for entry in scan():
        if entry.id == model_id:
            return entry.context
    return None


async def resident_ids(request: Request) -> frozenset[str]:
    """The ids the engine holds loaded — the one runtime fact a disk catalog needs. Async
    so it reads the engine's dict on the loop that mutates it."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return frozenset(engine.resident)


Resident = Annotated[frozenset[str], Depends(resident_ids)]

router = APIRouter()


def _find(model_id: str) -> CatalogEntry:
    for entry in scan():
        if entry.id == model_id:
            return entry
    raise HTTPException(status_code=404, detail=f"{model_id!r} is not in the catalog")


@router.get("/admin/models")
def models(loaded: Resident, resident: bool = False) -> list[CatalogEntry]:
    entries = [replace(entry, resident=entry.id in loaded) for entry in scan()]
    return [entry for entry in entries if entry.resident or not resident]


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int
    """Through the symlink: a hub snapshot is links into `blobs/`, and the listing answers
    for the bytes, not the link."""


@router.get("/admin/models/{model_id:path}/card", response_class=PlainTextResponse)
def card(model_id: str) -> str:
    """The checkpoint's README, raw — rendering it is the client's job."""
    path = _find(model_id).directory / "README.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{model_id!r} has no model card")
    return path.read_text()


@router.get("/admin/models/{model_id:path}/files")
def files(model_id: str) -> list[CheckpointFile]:
    directory = _find(model_id).directory
    return sorted(
        (
            CheckpointFile(name=path.name, size=path.stat().st_size)
            for path in directory.iterdir()
            if path.is_file()
        ),
        key=lambda file: file.name,
    )


@router.get("/admin/models/{model_id:path}/assets/{asset:path}")
def asset(model_id: str, asset: str) -> FileResponse:
    """A file the card references relatively (its images). The name resolves inside the
    checkpoint and nowhere else — `..` and absolute paths are refused before the disk is
    asked."""
    target = _find(model_id).directory / asset
    if asset.startswith("/") or ".." in PurePosixPath(asset).parts or not target.is_file():
        raise HTTPException(status_code=404, detail=f"{model_id!r} has no file {asset!r}")
    return FileResponse(target)


@router.get("/admin/models/{model_id:path}")
def model(model_id: str, loaded: Resident) -> CatalogEntry:
    return replace(_find(model_id), resident=model_id in loaded)


@router.delete("/admin/models/{model_id:path}", status_code=204)
def remove(model_id: str, loaded: Resident) -> None:
    entry = _find(model_id)
    if model_id in loaded:
        raise HTTPException(
            status_code=409,
            detail=f"{model_id!r} is resident: unload it before deleting it from disk",
        )
    shutil.rmtree(entry.store)
