"""What the catalog answers by id: the listing, the card, the files, the trace, the delete."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path, PurePosixPath

from mlx_omnia.engine.checkpoint import ImageCost
from mlx_omnia.engine.graph import Graph
from mlx_omnia.engine.graph import blueprint as trace_blueprint
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services.catalog.errors import (
    ImageSizeInvalid,
    ModelResident,
    NoModelCard,
    NoSuchAsset,
    NotTraceable,
    TakesNoImage,
    UnknownModel,
)
from mlx_omnia.server.services.catalog.scan import (
    CatalogEntry,
    CheckpointFile,
    entry_of,
    scan,
    sight_of,
)

Forget = Callable[[str], Awaitable[int]]
"""Everything the prefix tier stored for one model, dropped. Injected because whose spans
they are is the prefix service's business and not the catalog's."""


def resident_bytes(engine: Engine) -> Mapping[str, int | None]:
    """The ids the engine holds loaded, each with what its own tree says a decode step reads —
    the runtime facts a disk catalog does not have. Read on the loop that mutates it."""
    return {model_id: entry.active_bytes for model_id, entry in engine.residency.items()}


def _loaded(entry: CatalogEntry, resident: Mapping[str, int | None]) -> CatalogEntry:
    """A resident entry answers with the walk over the tree that is serving the requests, not
    with the estimate off the headers: a checkpoint ships blocks the loader drops and tensors
    it fuses, and the headers cannot know which."""
    if entry.id not in resident:
        return entry
    active = resident[entry.id]
    return replace(
        entry,
        resident=True,
        bytes_per_token=entry.bytes_per_token if active is None else active,
    )


def _find(model_id: str) -> CatalogEntry:
    entry = entry_of(model_id)
    if entry is None:
        raise UnknownModel(model_id)
    return entry


def models(
    resident: Mapping[str, int | None], *, only_resident: bool = False
) -> list[CatalogEntry]:
    entries = [_loaded(entry, resident) for entry in scan()]
    return [entry for entry in entries if entry.resident or not only_resident]


def model(model_id: str, resident: Mapping[str, int | None]) -> CatalogEntry:
    return _loaded(_find(model_id), resident)


def image_cost(model_id: str, height: int, width: int) -> ImageCost:
    """What one image of this size would cost this checkpoint, before it is sent: the size the
    tower actually reads, and the rows that reserves in the prompt.

    The arithmetic is the family's and not the dialect's — one resizes to a multiple of its
    patch block under a cap on area, the next under a cap on rows, the third tiles.
    """
    entry = _find(model_id)
    if height <= 0 or width <= 0:
        raise ImageSizeInvalid("an image has a positive height and width")
    eyes = sight_of(entry.architecture, entry.directory)
    if eyes is None:
        raise TakesNoImage(model_id)
    return eyes(height, width)


def card(model_id: str) -> str:
    """The checkpoint's README, raw — rendering it is the client's job."""
    path = _find(model_id).directory / "README.md"
    if not path.is_file():
        raise NoModelCard(model_id)
    return path.read_text()


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


@lru_cache(maxsize=16)
def _blueprint(directory: Path) -> Graph:
    """Cached per checkpoint: the tree a directory builds does not change while the files
    under it do not. The build reads no weight, so the cost is the shard headers and one
    uncomputed forward."""
    return trace_blueprint(directory)


def blueprint(model_id: str) -> Graph:
    """What this checkpoint's decode step is made of: the trunk, one graph per kind of block,
    and the kernel each declared operation resolved to.

    Recorded off the tree the loader builds rather than read off the config: nothing on disk
    says whether two mixers run side by side or one after the other. Nothing is loaded, which
    is why this asks the disk and not the engine's residency.
    """
    entry = _find(model_id)
    if not entry.supported:
        raise NotTraceable(f"no loader for {entry.architecture}: nothing to trace")
    try:
        return _blueprint(entry.directory)
    # Broad on purpose: a family's loader raises whatever its own transformation raises, and
    # what the reader needs is the sentence, not the class.
    except Exception as trouble:
        raise NotTraceable(f"{model_id!r} does not build: {trouble}") from trouble


def asset(model_id: str, asset_path: str) -> Path:
    """A file the card references relatively (its images). The name resolves inside the
    checkpoint and nowhere else — `..` and absolute paths are refused before the disk is
    asked."""
    target = _find(model_id).directory / asset_path
    if (
        asset_path.startswith("/")
        or ".." in PurePosixPath(asset_path).parts
        or not target.is_file()
    ):
        raise NoSuchAsset(f"{model_id!r} has no file {asset_path!r}")
    return target


async def remove(model_id: str, resident: Mapping[str, int | None], forget: Forget) -> None:
    """The weights and everything keyed to them. Unloading does not do this — surviving an
    unload is the whole point of the disk tier — but nothing keyed to a checkpoint outlives
    the checkpoint: what is left is bytes nobody can name."""
    entry = _find(model_id)
    if model_id in resident:
        raise ModelResident(f"{model_id!r} is resident: unload it before deleting it from disk")
    await asyncio.to_thread(shutil.rmtree, entry.store)
    await forget(model_id)
