"""The engine's window onto the daemon: config rows, the catalog and the prefix vault.

Blocking throughout, because the engine calls its environment from the loop, from worker
threads and from inside the decode loop — none of which the async pool serves. The reads go
through `db.sync_reads`, per call, which is the freshness the old paths had.
"""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

from mlx_omnia import LanguageModel, ModelInput, load
from mlx_omnia.engine.footprint import checkpoint_bytes
from mlx_omnia.engine.task import MTP_PREFIX
from mlx_omnia.server.db import sync_reads
from mlx_omnia.server.runtime.environment import Compression, DiskVault, Settings
from mlx_omnia.server.services import catalog, features
from mlx_omnia.server.services import config as config_service
from mlx_omnia.server.services.prefixes import FileVault


class _TextConfigJson(TypedDict):
    vocab_size: NotRequired[int]


class _ConfigJson(TypedDict):
    vocab_size: NotRequired[int]
    text_config: NotRequired[_TextConfigJson]


def _head_width(directory: Path) -> int | None:
    parsed: _ConfigJson = json.loads((directory / "config.json").read_text())
    width = parsed.get("vocab_size")
    text = parsed.get("text_config")
    return text.get("vocab_size") if width is None and text is not None else width


def config_now() -> config_service.Config:
    """The config in force, read per call the way the engine's paths always read it."""
    return config_service.Config.model_validate(
        {key: json.loads(value) for key, value in sync_reads.config_values().items()}
    )


class Daemon:
    """One instance per process, bound to the loop in the lifespan."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """The loop a vault reaches the database through. Set once, in the lifespan."""
        self._loop = loop

    def settings(self) -> Settings:
        current = config_now()
        return Settings(
            limit=current.memory_limit_bytes,
            ttl=current.idle_ttl_seconds,
            prefix_budget=current.prefix_cache_bytes,
            disk_budget=current.prefix_disk_bytes,
            span=current.prefix_span,
            not_resident=current.not_resident,
        )

    def concurrency(self, model_id: str) -> int:
        limit = config_now().max_concurrent_requests
        _, override = sync_reads.model_settings(model_id)
        return limit if override is None else min(limit, override)

    def incoming_bytes(self, model_id: str) -> int:
        entry = catalog.entry_of(model_id)
        if entry is None:
            return 0
        # The trunk's loader drops the MTP head; `draft_bytes` puts it back when the
        # settings turn speculation on, and counting it here as well would charge it twice.
        weights = checkpoint_bytes(entry.directory) - checkpoint_bytes(entry.directory, MTP_PREFIX)
        stored, _ = sync_reads.model_settings(model_id)
        return weights + features.draft_bytes(model_id, features.parse(stored).speculation)

    def head_width(self, model_id: str) -> int | None:
        entry = catalog.entry_of(model_id)
        return None if entry is None else _head_width(entry.directory)

    def kv_head_width(self, model_id: str) -> int | None:
        return catalog.kv_head_width(model_id)

    def stamp(self, model_id: str) -> str | None:
        return catalog.stamp_of(model_id)

    def vault(self, model_id: str, ceiling: int) -> DiskVault | None:
        loop = self._loop
        if loop is None or ceiling <= 0:
            return None
        return FileVault(loop, model_id, ceiling=ceiling)

    def compression(self, model_id: str) -> Compression | None:
        stored, _ = sync_reads.model_settings(model_id)
        return features.compression(features.parse(stored).kv_cache)


def resident_loader() -> Callable[[str], LanguageModel[ModelInput]]:
    """Only what is already on disk. Fetching a repository is a job of its own
    (`POST /admin/models`), and the catalog lists exactly what a client may name — so an id
    that is not there has to be an error rather than a download nobody asked for.

    The settings row is here for the model's own switches: whether a second checkpoint — the
    drafter — lands with it is a row.
    """

    def loader(model_id: str) -> LanguageModel[ModelInput]:
        model = load(model_id, local_files_only=True)
        stored, _ = sync_reads.model_settings(model_id)
        features.pair(model_id, model, features.parse(stored).speculation)
        return model

    return loader
