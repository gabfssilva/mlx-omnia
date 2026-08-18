"""The stand `test_residency` runs on: the weighted double, the slow one, and the app.

No checkpoint is opened. What these routes have to get right is where the last reference to a
model ends up, and a double holding one measurable `mx.array` is enough to see it.

The app is `create_app`'s and not a router mounted on a throwaway `FastAPI`: the residency
routes share `/admin/models` with the catalog's and the profiles' `{model_id:path}`, a
converter that matches slashes, so the mounting order is part of what every request here goes
through.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    KVCache,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
)
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Job, Loader
from mlx_omnia.server.services.jobs.registry import _WORKERS
from tests.server.conftest import app_of
from tests.server.engine_stand import caches_at

MODEL = "test/held"
"""The id carries a `/`, as every Hub id does: the `{model_id:path}` of these routes is what
has to survive it."""

_STORM = [f"test/storm-{index}" for index in range(_WORKERS + 2)]
"""Distinct ids, two more than either pool is wide: enough to fill the jobs pool and leave
work queued behind it, and enough to have filled the loop's own under the old design."""

_DEADLINE = 30.0
"""Every wait is bounded by it — a job that never lands fails here instead of hanging a suite
that has no global timeout."""

_WEIGHT_BYTES = 64 * 1024 * 1024
"""Large enough that `mx.get_active_memory()` moves by it unmistakably, small enough to be
loaded a few times over a suite."""

_ROWS = _WEIGHT_BYTES // (4 * 4)

_SLACK = 1024 * 1024
"""`mx.get_active_memory()` is the process's number, not this model's: anything else the suite
left alive is freed and taken under the reading, in both directions, and running this file
alone hides that — it passes there and fails in a full run. Measured drift was 2048 bytes on
the load and 2336 on the unload; a megabyte is three orders below the weights and still
separates the two readings this test exists to tell apart, since dropping only the dict key
leaves the delta at zero, not within a megabyte of 64 MiB."""


class Weights(nn.Module):
    """One embedding and nothing else: `resident_bytes` over this tree is `_WEIGHT_BYTES`."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(_ROWS, 4)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.embed(ids)


class Tokenizer:
    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        return iter([0])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"."


def weighted() -> CompositeModel[Text, Segment, GenerationOptions]:
    """The production chain down to the tree, materialized the way every loader's last line
    materializes it: a lazy tree occupies nothing, and occupying something is the point."""
    model = Weights()
    mx.eval(model.parameters())
    return CompositeModel(TextLanguageModel(model, Tokenizer()), [])


class SlowLanguageModel:
    """A generation slow enough to still be running when the DELETE arrives, and countable:
    ending intact means every token it promised, in order."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        for index in range(options.max_tokens):
            time.sleep(self.delay)
            yield Segment("content", str(index))


def slow(delay: float) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(SlowLanguageModel(delay), [])


@pytest.fixture(autouse=True)
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here reads the machine's own hub cache: a residency route mounted in the wrong
    order falls through to the catalog's, and the catalog scans what these two constants name.
    """
    caches_at(monkeypatch, tmp_path)


@asynccontextmanager
async def serving(loader: Loader) -> AsyncGenerator[tuple[httpx.AsyncClient, Engine]]:
    """The daemon's own app, run through its own lifespan: `ASGITransport` runs none, and the
    lifespan is what migrates the database, opens it, starts the jobs pool and the engine."""
    daemon = Daemon()
    engine = Engine(loader, daemon, Metrics())
    app = app_of(engine, daemon)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://residency") as http:
            yield http, engine


def url(model_id: str) -> str:
    return f"/admin/models/{model_id}/residency"


async def settled(http: httpx.AsyncClient, job_id: str) -> dict[str, object]:
    """The job as it ended, polled: a load that never finishes fails on the deadline."""
    deadline = time.monotonic() + _DEADLINE
    while True:
        response = await http.get(f"/admin/jobs/{job_id}")
        assert response.status_code == 200, response.text
        view = response.json()
        assert isinstance(view, dict)
        if view["state"] in {"ok", "error", "cancelled"}:
            return view
        assert time.monotonic() < deadline, f"the job stayed {view['state']!r}"
        await asyncio.sleep(0.005)


async def load(http: httpx.AsyncClient, model_id: str) -> dict[str, object]:
    """The PUT, and the job it hands back run to its end."""
    response = await asyncio.wait_for(http.put(url(model_id)), _DEADLINE)
    assert response.status_code == 202, response.text
    accepted = response.json()
    # The jobs screen filters by kind, so the string is a contract and not a label.
    assert accepted["kind"] == "load"
    assert response.headers["location"].endswith(accepted["id"])
    return await settled(http, accepted["id"])


async def unload(http: httpx.AsyncClient, model_id: str) -> httpx.Response:
    return await asyncio.wait_for(http.delete(url(model_id)), _DEADLINE)


async def resident(http: httpx.AsyncClient) -> list[dict[str, object]]:
    response = await asyncio.wait_for(http.get("/admin/state"), _DEADLINE)
    assert response.status_code == 200, response.text
    models = response.json()["models"]
    assert isinstance(models, list)
    return models


async def drain(job: Job) -> list[Segment]:
    """Bounded on purpose: a sentinel that stops being pushed has to fail the test rather
    than hang the suite."""
    pieces: list[Segment] = []
    while (piece := await asyncio.wait_for(job.chunks.get(), _DEADLINE)) is not None:
        pieces.append(piece)
    return pieces


async def occupied(http: httpx.AsyncClient, job_ids: list[str]) -> None:
    """Waits until the jobs pool is full, read off the rows: a body that has reported is a
    body that took a thread, and `report` is the first thing every one of them does.

    It is not the assertion the test is about — under one shared executor these same bodies
    take those threads too, and the wait passes there as well. What it buys is that the
    generation below is submitted with the pool provably full rather than at a guessed moment.
    """
    deadline = time.monotonic() + _DEADLINE
    while True:
        response = await asyncio.wait_for(http.get("/admin/jobs"), _DEADLINE)
        assert response.status_code == 200, response.text
        started = [
            view
            for view in response.json()
            if view["id"] in job_ids and view["progress"]["message"]
        ]
        if len(started) >= _WORKERS:
            return
        assert time.monotonic() < deadline, f"only {len(started)} load bodies ever started"
        await asyncio.sleep(0.01)
