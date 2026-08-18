"""Health, the daemon's own configuration, and what the machine underneath it is."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from mlx_omnia.server.api.management.common import STARTED, EngineDep, HostDep
from mlx_omnia.server.events import announce
from mlx_omnia.server.services import config as settings
from mlx_omnia.server.services import system as system_service

router = APIRouter()


@router.get("/admin/health")
async def health(engine: EngineDep) -> dict[str, object]:
    """The probe every client asks before anything else, and the one route an api key does not
    cover."""
    return {
        "status": "ok",
        "models": engine.resident,
        "pid": os.getpid(),
        "uptime": time.monotonic() - STARTED,
        "version": version("mlx_omnia"),
    }


Effect = Literal["applied", "restart", "inert"]

_EFFECTS: dict[str, Effect] = {
    "memory_limit_bytes": "applied",
    "idle_ttl_seconds": "applied",
    "max_concurrent_requests": "applied",
    "prefix_cache_bytes": "applied",
    "prefix_disk_bytes": "applied",
    "prefix_span": "applied",
    "port": "restart",
    "api_key": "applied",
    "catalog_directory": "inert",
    "not_resident": "applied",
}

_NOTES: dict[str, str] = {
    "catalog_directory": (
        "Kept, not honoured, and a restart does not honour it either: the scan reads"
        " `catalog.HUB_CACHE` while the loader resolves a checkpoint through"
        " `huggingface_hub`'s own constant, which is fixed from the environment at import."
        " Pointing this elsewhere would list checkpoints the daemon then cannot open."
    ),
}


@dataclass(frozen=True)
class Setting:
    """One parameter as the answer carries it. The effect is a property of the field and not
    of the value, so it comes back on the GET too."""

    value: int | str | None
    effect: Effect
    note: str | None = None


def _config_view(merged: settings.Config) -> dict[str, Setting]:
    return {
        name: Setting(value=value, effect=_EFFECTS[name], note=_NOTES.get(name))
        for name, value in merged.model_dump().items()
    }


@router.get("/admin/config")
async def config() -> dict[str, Setting]:
    return _config_view(await settings.current())


@router.patch("/admin/config")
async def update_config(
    body: settings.ConfigPatch, host: HostDep, request: Request
) -> dict[str, Setting]:
    """What is validated is the whole config and not the body: a bound belongs to the field,
    not to the request that happened to carry it. Only what the body named is written, so a
    field left alone goes on tracking its computed default."""
    sent = body.model_dump(exclude_unset=True)
    try:
        merged = settings.Config.model_validate((await settings.current()).model_dump() | sent)
    except ValidationError as error:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "body"
        raise HTTPException(status_code=400, detail=f"{field}: {first['msg']}") from error
    if merged.api_key is None and not settings.loopback(host):
        # The other half of `auth.check_bind`, which only runs at boot. Without this a daemon
        # that came up on the network *because* it had a key goes on serving without one the
        # moment somebody clears it.
        raise HTTPException(
            status_code=400,
            detail=f"api_key: this daemon is bound to {host}, where a key is what stands"
            " between the weights and the network. Bind 127.0.0.1 to serve without one.",
        )
    await settings.write(merged, set(sent))
    announce(request, "config")
    return _config_view(merged)


@router.get("/admin/system")
def system() -> system_service.SystemInfo:
    """Sync on purpose: the discovery is blocking IO, so FastAPI running it off the event loop
    is what keeps a generation in flight."""
    return system_service.system()
