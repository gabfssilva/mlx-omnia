"""The daemon's settings: nine parameters over a `config(key, value)` table.

Every value is stored as JSON rather than as its own string, because `None` is a value the
daemon means — no api key, no TTL — and a TEXT NOT NULL column has no room for a null.

A default is never written down. A field nobody has PATCHed has no row, and what the answer
reports is computed when it is asked: the ceiling is this machine's RAM minus 8 GB, the
catalog directory is the one the scan is using.
"""

import ipaddress
import json
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlx_omnia.engine.core.prefix import SPAN
from mlx_omnia.server.db import base
from mlx_omnia.server.db.models.profiles import Config as ConfigRow
from mlx_omnia.server.services import catalog

_RESERVED_BYTES = 8 * 1024**3

Policy = Literal["load", "fail"]


def loopback(host: str) -> bool:
    """Whether a bind to this host is reachable only from the machine itself. A name that is
    not an address — `localhost` aside — counts as off the loopback: it resolves to whatever
    the resolver says, and the conservative reading of that is the one that asks for a key."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _memory_ceiling() -> int:
    """This machine's RAM minus what the rest of it needs. Read rather than assumed: a
    ceiling above the installed memory is a ceiling that never evicts."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") - _RESERVED_BYTES


_PREFIX_SHARE = 32
"""One part in `_PREFIX_SHARE` of the memory ceiling is what the resident models may keep of
prefixes by default, between them."""

_PREFIX_FLOOR = 1024**3
"""Never less than the gibibyte that was the flat default before."""


class Config(BaseModel):
    """The nine and their bounds. This is also where a PATCH is judged: reading a row is a
    validation and not a cast, so a value hand-edited into the file reaches nothing."""

    memory_limit_bytes: int = Field(default_factory=_memory_ceiling, gt=0)
    idle_ttl_seconds: int | None = Field(default=1800, gt=0)
    """`None` is never — the high TTL standing in for pinning."""
    max_concurrent_requests: int = Field(default=1, ge=1)
    prefix_cache_bytes: int = Field(default=_PREFIX_FLOOR, ge=0)
    """How much the resident models may keep, together, of the prefixes they have already
    read, 0 for none. Counted inside `memory_limit_bytes`, so the default is a share of that
    ceiling, filled in by the validator below and never written to the file."""
    prefix_disk_bytes: int = Field(default=8 * 1024**3, ge=0)
    """How much of the prefix directory the daemon may fill with conversations the memory
    trie evicted, 0 for none. Across all models: it is one disk."""
    prefix_span: int = Field(default=SPAN, ge=64, le=4096, multiple_of=64)
    """Tokens per span — the granularity a conversation is stored and resumed at. It rides
    in the key, so moving it costs one prefill and corrupts nothing."""
    port: int = Field(default=8642, ge=1, le=65535)
    api_key: str | None = None
    catalog_directory: str = Field(default_factory=lambda: str(catalog.HUB_CACHE))
    not_resident: Policy = "load"
    """What a request naming a model that is not loaded gets: the load, or the error."""

    @model_validator(mode="before")
    @classmethod
    def _share_the_ceiling_with_the_trie(cls, values: object) -> object:
        """`prefix_cache_bytes` follows the ceiling it is counted inside, including a ceiling
        the user PATCHed down. Before and not after, because the point is to fill a field
        nobody wrote: a value that arrived is a value the user chose."""
        if not isinstance(values, dict) or "prefix_cache_bytes" in values:
            return values
        limit = values.get("memory_limit_bytes")
        ceiling = limit if isinstance(limit, int) else _memory_ceiling()
        return values | {"prefix_cache_bytes": max(_PREFIX_FLOOR, ceiling // _PREFIX_SHARE)}


class ConfigPatch(BaseModel):
    """Types only — the bounds live on `Config`, which is what the merged config is checked
    against. `None` is a value here (it clears the api key and the TTL), which is why what
    the body carried is `model_fields_set` and never the absence of a value."""

    model_config = ConfigDict(extra="forbid")

    memory_limit_bytes: int | None = None
    idle_ttl_seconds: int | None = None
    max_concurrent_requests: int | None = None
    prefix_cache_bytes: int | None = None
    prefix_disk_bytes: int | None = None
    prefix_span: int | None = None
    port: int | None = None
    api_key: str | None = None
    catalog_directory: str | None = None
    not_resident: Policy | None = None


async def current() -> Config:
    """The config in force. Nothing caches it; a key the model no longer declares is ignored
    rather than fatal, because a leftover row from another version would otherwise make
    every read fail."""
    rows = await ConfigRow.objects.all()
    return Config.model_validate({row.key: json.loads(row.value) for row in rows})


async def write(merged: Config, names: set[str]) -> None:
    """Only what the body named is written, so a field left alone goes on tracking its
    computed default — and the keys of one PATCH land together or not at all."""
    values = merged.model_dump()
    async with base.database.transaction():
        for name in names:
            value = json.dumps(values[name])
            if not await ConfigRow.objects.filter(key=name).update(value=value):
                await ConfigRow(key=name, value=value).save()
