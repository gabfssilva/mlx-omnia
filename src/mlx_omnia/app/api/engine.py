"""What the window knows about the machine, and every resource it draws.

The daemon's own contract and nothing beside it — `/admin/system` once, then one
`/admin/events` stream carrying whole resources — so this app is a client of the daemon
and not a second opinion about it. Shapes are the server's; where a field is
optional there it is `| None` here, because FastAPI serialises None rather than dropping
the key.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal, TypedDict, cast
from urllib.parse import quote

import flet as ft
import httpx

from mlx_omnia.app.api.http import BASE, events, get, send

# A daemon that is down is a daemon somebody is restarting, and a client reconnecting on
# a tight loop is the load the restart does not need.
REDIAL = 1.0

GIB = 1024**3


class SystemInfo(TypedDict):
    chip: str
    gpu_cores: int
    memory_bytes: int
    bandwidth_theoretical_gbs: float
    bandwidth_sustained_gbs: float
    disk_free_bytes: int
    catalog: str
    version: str
    constants: dict[str, str]


class Health(TypedDict):
    status: str
    models: list[str]
    pid: int
    uptime: float
    version: str


class Slot(TypedDict):
    id: str
    weights_bytes: int
    kv_bytes: int
    loaded_at: float
    last_used: float | None


class Queue(TypedDict):
    running: int
    waiting: int


class State(TypedDict):
    models: list[Slot]
    queue: Queue
    resident_bytes: int
    kv_bytes: int


class Setting(TypedDict):
    value: float | str | None
    effect: Literal["applied", "restart", "inert"]
    note: str | None


class SamplingDefaults(TypedDict):
    """What the checkpoint's own `generation_config.json` declares. A knob it does not
    declare is null, which is not the same as one it declares neutral."""

    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None


class CatalogEntry(TypedDict):
    id: str
    defaults: SamplingDefaults
    directory: str
    # What the id owns on disk, and what DELETE removes.
    store: str
    architecture: str
    quantization: str | None
    dtype: str | None
    context: int | None
    bytes_on_disk: int
    bytes_per_token: float | None
    # What the cache grows by per token, over every attending layer. `null` is a config
    # that does not answer, never a guess.
    kv_bytes_per_token: float | None
    # What fidelity pairs are decided by: the same vocabulary is what makes two logit
    # vectors subtractable, and the same shape is what makes the difference mean something.
    vocab_size: int | None
    shape: str | None
    resident: bool
    # Whether the engine has a loader for the architecture — what a picker may offer to
    # run. Residency itself is the state stream's to say: this list is republished on
    # catalog changes, not on loads.
    supported: bool


class Progress(TypedDict):
    message: str
    completed: int
    total: int | None


class Job(TypedDict):
    id: str
    kind: Literal["download", "load", "quantize", "bench", "benchmark"]
    subject: dict[str, str | list[str]]
    state: Literal["pending", "running", "ok", "error", "cancelled"]
    progress: Progress
    updated_at: float
    error: str | None


class Speculation(TypedDict):
    """What a drafter proposed and how much of it the target confirmed."""

    rounds: int
    proposed: int
    accepted: int


class Sample(TypedDict):
    model: str
    state: Literal["queued", "running", "completed", "cancelled", "error"]
    prompt_tokens: int
    completion_tokens: int
    started_at: float
    # The seconds this request spent putting its model in memory, absent when it found it
    # resident. It is outside `ttft` — the meter starts at the prefill.
    load_seconds: float | None
    ttft: float | None
    tokens_per_second: float | None
    prefill_tokens_per_second: float | None
    ceiling_fraction: float | None
    speculation: Speculation | None


class Aggregate(TypedDict):
    model: str
    requests: int
    tokens_per_second: float | None
    ceiling_fraction: float | None


class Snapshot(TypedDict):
    live: Sample | None
    requests: list[Sample]
    models: list[Aggregate]


@ft.observable
@dataclass
class Engine:
    """Every resource the window draws, as of the last frame.

    Observable: a component that takes it as an argument is subscribed to it, so writing a
    resource here is what redraws the screens reading that resource — there is no repaint
    loop, and a frame nobody reads costs nothing.
    """

    system: SystemInfo | None = None
    state: State | None = None
    health: Health | None = None
    config: dict[str, Setting] = field(default_factory=dict)
    catalog: list[CatalogEntry] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    metrics: Snapshot | None = None

    @property
    def models(self) -> list[Slot]:
        """Resident, biggest first — the order the band and the roster are drawn in."""
        if self.state is None:
            return []
        return sorted(
            self.state["models"], key=lambda m: -(m["weights_bytes"] + m["kv_bytes"])
        )

    @property
    def materials(self) -> dict[str, int]:
        """One material per resident model, by load order: colour is identity, so a block
        does not change colour because another one left."""
        if self.state is None:
            return {}
        by_age = sorted(self.state["models"], key=lambda m: m["loaded_at"])
        return {model["id"]: index for index, model in enumerate(by_age)}

    @property
    def ceiling(self) -> int | None:
        limit = self.config.get("memory_limit_bytes")
        value = None if limit is None else limit["value"]
        return int(value) if isinstance(value, int | float) else None

    def entry(self, model: str) -> CatalogEntry | None:
        return next((item for item in self.catalog if item["id"] == model), None)

    def aggregate(self, model: str) -> Aggregate | None:
        if self.metrics is None:
            return None
        return next((a for a in self.metrics["models"] if a["model"] == model), None)

    def live(self, model: str) -> Sample | None:
        sample = None if self.metrics is None else self.metrics["live"]
        return sample if sample is not None and sample["model"] == model else None


async def follow(engine: Engine) -> None:
    """Fill `engine` from the daemon for as long as this task lives.

    The machine does not change while the app runs, so it is the one thing still fetched;
    everything else is pushed. Each frame names the resource it carries — anything else is
    a daemon newer than this window, and ignoring it is what lets one add a resource
    without breaking the other.
    """
    async with httpx.AsyncClient(base_url=BASE, timeout=None) as client:
        while True:
            try:
                if engine.system is None:
                    answer = await client.get("/admin/system")
                    answer.raise_for_status()
                    engine.system = cast(SystemInfo, answer.json())
                async for frame in events(client, "/admin/events"):
                    _apply(engine, frame)
            except (httpx.HTTPError, json.JSONDecodeError):
                # The daemon went away, or has not come up yet. Which one it was is the
                # next dial's to find out.
                pass
            engine.health = None
            await asyncio.sleep(REDIAL)


def _apply(engine: Engine, frame: dict[str, object]) -> None:
    resource, value = frame.get("resource"), frame.get("value")
    match resource:
        case "health":
            engine.health = cast(Health, value)
        case "config":
            engine.config = cast(dict[str, Setting], value)
        case "models":
            engine.catalog = cast(list[CatalogEntry], value)
        case "state":
            engine.state = cast(State, value)
        case "jobs":
            engine.jobs = cast(list[Job], value)
        case "metrics":
            engine.metrics = cast(Snapshot, value)
        case _:
            pass


# ── the rest of the daemon, called rather than streamed ───────────────────


async def job_events(job: str) -> AsyncGenerator[Job]:
    """Every frame is the whole job and the stream opens with its state as it is now, so a
    late subscriber loses nothing. It ends when the job reaches a terminal state."""
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as http:
        async for frame in events(http, f"/admin/jobs/{job}/events"):
            yield cast(Job, frame)


async def active_jobs() -> list[Job]:
    return cast(list[Job], await get("/admin/jobs?active=true"))


async def cancel_job(job: str) -> None:
    await send("DELETE", f"/admin/jobs/{job}")


async def unload(model: str) -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        answer = await client.delete(f"/admin/models/{quote(model, safe='')}/residency")
        answer.raise_for_status()


async def patch_config(patch: dict[str, object]) -> None:
    """An absent key and a null one are different requests, so the caller sends exactly
    the fields it means to write."""
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        answer = await client.patch("/admin/config", json=patch)
        answer.raise_for_status()
