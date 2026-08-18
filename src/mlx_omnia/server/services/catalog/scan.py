"""The scan: what the caches hold, and the numbers each listed entry is a function of."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from stat import S_ISREG

import huggingface_hub.constants

from mlx_omnia.engine.checkpoint import SamplingDefaults, Sight, sampling_defaults
from mlx_omnia.engine.task import architectures, sight
from mlx_omnia.server.services.catalog import headers, kv
from mlx_omnia.server.services.catalog.config import (
    ConfigJson,
    print_of,
    quantization_of,
    shape_of,
)

HUB_CACHE = Path(huggingface_hub.constants.HF_HUB_CACHE)
QUANTIZED_CACHE = Path.home() / ".cache" / "mlx_omnia" / "quantized"


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    directory: Path
    """The checkpoint: `config.json` and the shards."""
    store: Path
    """What the id owns on disk, and what a delete removes. A hub repository owns its whole
    folder: the snapshot is symlinks into `blobs/`, and unlinking it alone frees nothing."""
    architecture: str
    quantization: str | None
    dtype: str | None
    context: int | None
    bytes_on_disk: int
    defaults: SamplingDefaults = field(default_factory=SamplingDefaults)
    """How the checkpoint's own `generation_config.json` says it wants to be sampled."""
    bytes_per_token: int | None = None
    """Bytes a decode step reads, and the denominator of every "% of the ceiling". `None`
    when the shards' headers cannot be read."""
    kv_bytes_per_token: int | None = None
    """What the cache grows by per token, over every attending layer."""
    attention_window: int | None = None
    """Where the cache stops growing. `None` for full attention, and also for a checkpoint
    that only slides on some of its layers."""
    vocab_size: int | None = None
    shape: str | None = None
    """A print of the architecture. It validates a declared fidelity pair; it never invents
    kinship, since a fine-tune's print is identical to its base's."""
    resident: bool = False
    """Filled from the engine, not by the scan."""
    supported: bool = False
    """Whether this engine has a loader for the architecture. A tokenizer can still refuse at
    load time."""
    sees: bool = False
    """Whether a turn may carry an image. Per checkpoint and not per architecture."""


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int
    """Through the symlink: a hub snapshot is links into `blobs/`."""


_Stamp = tuple[tuple[str, int, int], ...]


def stamp_of(model_id: str) -> str | None:
    """One string that moves when the checkpoint under an id does, or `None` for an id the
    scan does not list. It keys anything derived from a checkpoint's own numbers: a file
    written under an id and read back after a shard was replaced is the wrong answer, arriving
    without an error."""
    entry = entry_of(model_id)
    if entry is None:
        return None
    digest = hashlib.sha256()
    for name, size, mtime in _stamp(Path(entry.directory)):
        digest.update(f"{name}\0{size}\0{mtime}\0".encode())
    return digest.hexdigest()


def _stamp(directory: Path) -> _Stamp:
    """Every file the entry is a function of, by name, size and mtime. `stat` follows a
    snapshot's links into `blobs/`, so a shard landing moves the stamp."""
    found: list[tuple[str, int, int]] = []
    for path in sorted(directory.iterdir()):
        try:
            info = path.stat()
        except OSError:
            continue
        if S_ISREG(info.st_mode):
            found.append((path.name, info.st_size, info.st_mtime_ns))
    return tuple(found)


@lru_cache(maxsize=256)
def _build(model_id: str, directory: Path, store: Path, stamp: _Stamp) -> CatalogEntry | None:
    """Keyed by the stamp, which is what every number below is read from. `None` is cached
    too — an interrupted download is re-listed as often as a complete one, and it stops being
    `None` when its last shard moves the stamp."""
    config_path = directory / "config.json"
    if not config_path.is_file() or not headers.complete(directory):
        return None
    config: ConfigJson = json.loads(config_path.read_text())
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
        quantization=quantization_of(config),
        dtype=config.get("dtype") or config.get("torch_dtype"),
        context=context,
        defaults=sampling_defaults(directory),
        bytes_on_disk=sum(size for _, size, _ in stamp),
        bytes_per_token=headers.bytes_per_token(directory, config),
        kv_bytes_per_token=kv.bytes_per_token(directory, config),
        attention_window=kv.attention_window(shape_of(config)),
        vocab_size=shape_of(config).get("vocab_size"),
        shape=print_of(config),
        supported=architecture in architectures(),
        sees=sight_of(architecture, directory) is not None,
    )


def sight_of(architecture: str, directory: Path) -> Sight | None:
    """The engine's answer, and a listing that survives it not having one: a checkpoint whose
    config mirror is refused is still a checkpoint to list, with no picture offered over it."""
    try:
        return sight(architecture, directory)
    except Exception:  # an unreadable config is a model that takes no image
        return None


def _entry(model_id: str, directory: Path, store: Path) -> CatalogEntry | None:
    return _build(model_id, directory, store, _stamp(directory))


def _head(repository: Path) -> Path | None:
    """The revision `load` would resolve. A snapshot fetched by sha has no `refs/main`, so the
    fallback is the most recent one rather than nothing."""
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
            # A quantization stages under `.tmp-*` next to the entry it is about to become.
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            if (entry := _entry(str(directory), directory, directory)) is not None:
                yield entry


def scan() -> list[CatalogEntry]:
    return sorted([*_hub(HUB_CACHE), *_quantized(QUANTIZED_CACHE)], key=lambda entry: entry.id)


def entry_of(model_id: str) -> CatalogEntry | None:
    """The scanned entry under an id, or `None` for an id the disk does not answer for.

    Not cached, and deliberately: the scan behind it is what notices a checkpoint that arrived
    or left, and every caller here wants this instant's answer."""
    return next((entry for entry in scan() if entry.id == model_id), None)


@lru_cache(maxsize=256)
def context_of(model_id: str) -> int | None:
    """The `max_position_embeddings` the checkpoint declares, by id — what a dialect caps a
    generation with. Cached because it rides the request path and a checkpoint's context does
    not move."""
    entry = entry_of(model_id)
    return None if entry is None else entry.context


@lru_cache(maxsize=256)
def defaults_of(model_id: str) -> SamplingDefaults:
    """The checkpoint's sampling defaults, by id. Empty for an id the disk does not answer
    for, which is the same thing as a checkpoint that declares nothing."""
    entry = entry_of(model_id)
    return SamplingDefaults() if entry is None else entry.defaults


def kv_head_width(model_id: str) -> int | None:
    """The width a compressed KV cache's formats have to close their groups on. A
    `QuantizedKVCache` packs along the head and never along tokens, so the axis is one row of
    one layer. `None` when the config does not answer."""
    entry = entry_of(model_id)
    if entry is None:
        return None
    config: ConfigJson = json.loads((entry.directory / "config.json").read_text())
    return kv.head_width(shape_of(config))
