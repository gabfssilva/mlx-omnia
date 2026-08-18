"""What is resident right now, the prefix tiers, and the two streams the app watches."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

import mlx.core as mx
from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from mlx_omnia.server.api import sse
from mlx_omnia.server.api.management.common import (
    KEEP_ALIVE_SECONDS,
    EngineDep,
    HubDep,
    MetricsDep,
    Tier,
)
from mlx_omnia.server.events import Hub, announce
from mlx_omnia.server.metrics import Metrics, Snapshot
from mlx_omnia.server.runtime.environment import KvCompression
from mlx_omnia.server.runtime.footprint import footprint_bytes
from mlx_omnia.server.services import prefixes

router = APIRouter()


@dataclass(frozen=True)
class Resident:
    id: str
    weights_bytes: int
    kv_bytes: int
    loaded_at: float
    last_used: float | None
    kv_cache: KvCompression | None = None
    """The verdict on this model's compressed-KV policy, `None` while its settings ask for
    none."""


@dataclass(frozen=True)
class Queue:
    running: int
    waiting: int
    reserved: bool = False
    """Somebody is holding the queue exclusively — a benchmark, today."""


@dataclass(frozen=True)
class State:
    models: list[Resident]
    queue: Queue
    resident_bytes: int
    """`max(MLX's active memory, the process' resident size, the accumulator)`, the accumulator
    being the sum of `models[*].weights_bytes`."""
    kv_bytes: int
    prefix_memory_bytes: int
    prefix_disk_bytes: int


@router.get("/admin/state")
async def state(engine: EngineDep) -> State:
    models = [
        Resident(
            id=model_id,
            weights_bytes=entry.weights_bytes,
            kv_bytes=entry.kv_bytes,
            loaded_at=entry.loaded_at,
            last_used=entry.last_used,
            kv_cache=entry.kv_cache,
        )
        for model_id, entry in engine.residency.items()
    ]
    accumulator = sum(model.weights_bytes for model in models)
    return State(
        models=models,
        queue=Queue(running=engine.running, waiting=engine.waiting, reserved=engine.reserved),
        resident_bytes=max(mx.get_active_memory(), footprint_bytes(), accumulator),
        kv_bytes=sum(model.kv_bytes for model in models),
        prefix_memory_bytes=engine.prefix_bytes,
        prefix_disk_bytes=await prefixes.total_bytes(),
    )


@router.delete("/admin/prefixes/{tier}", status_code=204)
async def clear_prefixes(tier: Tier, engine: EngineDep, request: Request) -> None:
    """Hand one floor of the prefix cache back.

    The memory tier is discarded and never drained: a clear that spilled would answer "free
    this memory" by handing the disk tier what was just freed.
    """
    if tier == "memory":
        engine.discard_prefixes()
    else:
        for model_id in {row.model for row in await prefixes.rows()}:
            await prefixes.forget(model_id)
    announce(request, "state")


def _resource_frame(resource: str, value: object) -> str:
    return f"data: {json.dumps({'resource': resource, 'value': jsonable_encoder(value)})}\n\n"


async def _resource_frames(hub: Hub) -> AsyncIterator[str]:
    """Subscribed before the first resource is read, so a change that lands during the opening
    burst is delivered after it instead of being missed. It ends when the client goes away and
    never on its own — there is no last frame, the way there is no last state."""
    with hub.watch() as watcher:
        for resource, source in hub.sources.items():
            yield _resource_frame(resource, await source())
        while True:
            try:
                await asyncio.wait_for(watcher.wake.wait(), KEEP_ALIVE_SECONDS)
            except TimeoutError:
                yield sse.KEEP_ALIVE
                continue
            watcher.wake.clear()
            pending, watcher.dirty = watcher.dirty, set()
            for resource in sorted(pending):
                yield _resource_frame(resource, await hub.sources[resource]())


@router.get("/admin/events")
async def events(hub: HubDep) -> StreamingResponse:
    return StreamingResponse(_resource_frames(hub), media_type=sse.MEDIA_TYPE)


def _snapshot_frame(snapshot: Snapshot) -> str:
    return f"data: {json.dumps(asdict(snapshot))}\n\n"


async def _snapshot_frames(register: Metrics) -> AsyncIterator[str]:
    with register.watch() as queue:
        current = register.snapshot()
        yield _snapshot_frame(current)
        while True:
            try:
                current = await asyncio.wait_for(queue.get(), KEEP_ALIVE_SECONDS)
            except TimeoutError:
                latest = register.snapshot()
                if latest == current:
                    yield sse.KEEP_ALIVE
                    continue
                current = latest
            yield _snapshot_frame(current)


@router.get("/admin/metrics")
async def metrics(register: MetricsDep) -> Snapshot:
    return register.snapshot()


@router.get("/admin/metrics/events")
async def metrics_events(register: MetricsDep) -> StreamingResponse:
    return StreamingResponse(_snapshot_frames(register), media_type=sse.MEDIA_TYPE)
