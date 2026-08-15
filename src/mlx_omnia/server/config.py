"""GET/PATCH /admin/config: the nine parameters the Server screen edits.

The table under it is `config(key, value)`, both TEXT (35.1); the typing over it is here.
Every value is stored as JSON rather than as its own string, because `None` is a value the
daemon means — no api key, no TTL — and a TEXT NOT NULL column has no room for a null.
`json.dumps` gives them one encoding and one way back.

A default is never written down. A field nobody has PATCHed has no row, and what the answer
reports is computed when it is asked: the ceiling is this machine's RAM minus 8 GB, the
catalog directory is the one the scan is using. Writing the whole config on the first PATCH
would freeze one machine's RAM into a file that gets copied to the next one.

What the answer says **per field**, rather than in a footnote under all of them, is when a
change starts counting:

- `applied` — nothing caches the config, so what a PATCH leaves is what the next call to
  `current()` returns. The ceiling and the TTL are of this kind; the sweep that reads them
  is 32.5's.
- `restart` — the process fixed it when it came up. The port is the whole of this one: `main`
  reads it here when no `--port` says otherwise, and the socket is bound before anything is
  served.
- `inert` — kept, and acted on by nothing, which the note beside it has to say.
  `catalog_directory` is of this kind: a restart would not honour it either — the scan reads
  `catalog.HUB_CACHE` while the
  loader resolves through `huggingface_hub`'s own constant, fixed from the environment at
  import. An API answering `applied` or `restart` for either would be telling the user the
  opposite of what happens while taking the value in silence.
"""

import ipaddress
import json
import os
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mlx_omnia.server import catalog
from mlx_omnia.server.deps import StoreDep
from mlx_omnia.server.events import announce
from mlx_omnia.server.store import Store

_RESERVED_BYTES = 8 * 1024**3

Policy = Literal["load", "fail"]
Effect = Literal["applied", "restart", "inert"]


def loopback(host: str) -> bool:
    """Whether a bind to this host is reachable only from the machine itself. A name that is
    not an address — `localhost` aside — counts as off the loopback: it resolves to whatever
    the resolver says, and the conservative reading of that is the one that asks for a key.

    It lives here rather than in `auth` because it is a rule about the configuration, and
    both readers are on this side of the import: `auth.check_bind` at boot, and the PATCH
    below, which is the other half of the same guarantee."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _memory_ceiling() -> int:
    """This machine's RAM minus what the rest of it needs. Read rather than assumed: the
    same daemon ships to a 36 GB and to a 128 GB Mac, and a ceiling above the installed
    memory is a ceiling that never evicts."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") - _RESERVED_BYTES


_PREFIX_SHARE = 32
"""One part in `_PREFIX_SHARE` of the memory ceiling is what the resident models may keep of
prefixes by default, between them.

The measurement behind it, on the checkpoint that raised the question: `Ling-3.0-flash`
costs 8 064 bytes of latent cache per token over its 7 attending layers, so a 62k-token
conversation is ~500 MB, plus ~73 MB of recurrent state that is fixed and does not scale
with the context — ~573 MB for the entry the trie holds. A gibibyte therefore held one
conversation and evicted the second, at a whole prefill each time, which is the failure this
number exists to move: a share of 32 is 3.75 GB on a 128 GB Mac, six of those conversations.
"""

_PREFIX_FLOOR = 1024**3
"""Never less than the gibibyte that was the flat default before. One part in 32 of a 36 GB
Mac's ceiling is 875 MB, and scaling up is not a reason to take memory away from the small
machine."""


class Config(BaseModel):
    """The nine and their bounds. This is also where a PATCH is judged: reading a row is a
    validation and not a cast, so a value hand-edited into the file reaches nothing."""

    memory_limit_bytes: int = Field(default_factory=_memory_ceiling, gt=0)
    idle_ttl_seconds: int | None = Field(default=1800, gt=0)
    """`None` is never — the high TTL decision 3 leaves standing in for pinning."""
    max_concurrent_requests: int = Field(default=1, ge=1)
    prefix_cache_bytes: int = Field(default=_PREFIX_FLOOR, ge=0)
    """How much the resident models may keep, together, of the prefixes they have already
    read, 0 for none.

    Counted inside `memory_limit_bytes`, because the trie's arrays are live allocations the
    memory rail already reads — so the default is a share of that same ceiling rather than a
    constant beside it, filled in by the validator below and never written to the file.

    Across all of them, like `prefix_disk_bytes`: it is one machine's memory, and split per
    model the same number would mean twice as much with two resident and half as much
    available to one. Each model still holds its own trie — the element type is the trunk's
    own cache, and it dies with the model, which is what keeps one prompt in two resident
    models from crossing caches — and the accountant that weighs them against each other is
    `prompt_cache.Budget`, on the engine.
    """
    prefix_disk_bytes: int = Field(default=8 * 1024**3, ge=0)
    """How much of `~/.cache/mlx_omnia/prefixes` the daemon may fill with conversations the
    memory trie evicted, 0 for none. Across all models, like `prefix_cache_bytes` — this is a
    disk and there is one of those, and the LRU that enforces it evicts by last use, so two
    models compete for it the way two conversations of one model do.

    Eight gibibytes is about fourteen conversations of the length that raised this — 573 MB
    each on `Ling-3.0-flash` at 62k tokens — and a flat number rather than a share of the
    disk, because a free disk is a fact about the machine at this instant and a ceiling that
    moved with it would grow into whatever was just freed.

    What it buys is the turn after a restart: the trie dies with the process, so without this
    every conversation that was warm pays a whole prefill again."""
    port: int = Field(default=8642, ge=1, le=65535)
    api_key: str | None = None
    catalog_directory: str = Field(default_factory=lambda: str(catalog.HUB_CACHE))
    not_resident: Policy = "load"
    """What a request naming a model that is not loaded gets: the load, or the error
    (decision 3)."""

    @model_validator(mode="before")
    @classmethod
    def _share_the_ceiling_with_the_trie(cls, values: object) -> object:
        """`prefix_cache_bytes` follows the ceiling it is counted inside, including a ceiling
        the user PATCHed down — a fixed share of a limit somebody set to 8 GB is the honest
        reading of "a share", and a constant beside it would be nearly half of it.

        Before and not after, because the point is to fill a field nobody wrote: a value that
        arrived is a value the user chose, and the merge in `update` always carries one.
        """
        if not isinstance(values, dict) or "prefix_cache_bytes" in values:
            return values
        limit = values.get("memory_limit_bytes")
        ceiling = limit if isinstance(limit, int) else _memory_ceiling()
        return values | {"prefix_cache_bytes": max(_PREFIX_FLOOR, ceiling // _PREFIX_SHARE)}


class ConfigPatch(BaseModel):
    """Types only — the bounds live on `Config`, which is what the merged config is checked
    against, so there is one place a limit is written down. `None` is a value here (it clears
    the api key and the TTL), which is why what the body carried is `model_fields_set` and
    never the absence of a value."""

    model_config = ConfigDict(extra="forbid")

    memory_limit_bytes: int | None = None
    idle_ttl_seconds: int | None = None
    max_concurrent_requests: int | None = None
    prefix_cache_bytes: int | None = None
    prefix_disk_bytes: int | None = None
    port: int | None = None
    api_key: str | None = None
    catalog_directory: str | None = None
    not_resident: Policy | None = None


_EFFECTS: dict[str, Effect] = {
    "memory_limit_bytes": "applied",
    "idle_ttl_seconds": "applied",
    "max_concurrent_requests": "applied",
    # Read per request in `Engine.submit` and carried into the model, which rebuilds its trie
    # when the number moves — so a PATCH counts for the next request on a model already up.
    "prefix_cache_bytes": "applied",
    # Read per request too, and by the same path: the spill the engine hands down carries the
    # ceiling it was built with, and the engine rebuilds it when the number moves.
    "prefix_disk_bytes": "applied",
    "port": "restart",
    # `auth.ApiKey` reads the store per request, so a rotation — and a revocation, which is
    # the one that matters — is in force on the next call.
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
    of the value, so it comes back on the GET too — the screen draws the warning beside the
    control, before anyone edits it."""

    value: int | str | None
    effect: Effect
    note: str | None = None


def current(store: Store) -> Config:
    """The config in force. Nothing caches it, which is what makes `applied` above true; a
    key the model no longer declares is ignored rather than fatal, because a leftover row
    from another version would otherwise make every read of this route fail."""
    return Config.model_validate({key: json.loads(raw) for key, raw in store.config().items()})


def _view(config: Config) -> dict[str, Setting]:
    values = config.model_dump()
    return {
        name: Setting(value=value, effect=_EFFECTS[name], note=_NOTES.get(name))
        for name, value in values.items()
    }


def _host(request: Request) -> str:
    host = request.app.state.host
    assert isinstance(host, str)
    return host


HostDep = Annotated[str, Depends(_host)]

router = APIRouter()


@router.get("/admin/config")
def config(store: StoreDep) -> dict[str, Setting]:
    return _view(current(store))


@router.patch("/admin/config")
def update(
    body: ConfigPatch, store: StoreDep, host: HostDep, request: Request
) -> dict[str, Setting]:
    """What is validated is the whole config and not the body: a bound belongs to the field,
    not to the request that happened to carry it. Only what the body named is written, so a
    field left alone goes on tracking its computed default — and `set_config` is one
    `executemany` in one transaction, so the four fields of a PATCH land together, while a
    merge that fails a bound writes nothing at all.
    """
    sent = body.model_dump(exclude_unset=True)
    try:
        merged = Config.model_validate(current(store).model_dump() | sent)
    except ValidationError as error:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "body"
        raise HTTPException(status_code=400, detail=f"{field}: {first['msg']}") from error
    if merged.api_key is None and not loopback(host):
        # The other half of `auth.check_bind`, which only runs at boot. Without this a daemon
        # that came up on the network *because* it had a key goes on serving without one the
        # moment somebody clears it — every route but `/admin/health` open, including the one
        # that deletes checkpoints, and no restart to notice it at.
        raise HTTPException(
            status_code=400,
            detail=f"api_key: this daemon is bound to {host}, where a key is what stands"
            " between the weights and the network. Bind 127.0.0.1 to serve without one.",
        )
    values = merged.model_dump()
    store.set_config({name: json.dumps(values[name]) for name in sent})
    announce(request, "config")
    return _view(merged)
