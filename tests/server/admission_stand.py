"""The stand `test_admission` runs on: the weighted double, the fake hub, and the app.

No checkpoint is opened, but one is *written*. Admission sizes what it is about to load off
the safetensors headers — the only figure that exists before the load — so the fake cache
here carries a header that declares the bytes with no payload behind it, and the double the
loader hands back is what actually occupies them.
"""

import asyncio
import json
import threading
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
from mlx_omnia.server.runtime.footprint import footprint_bytes
from tests.server.conftest import app_of
from tests.server.engine_stand import caches_at

FIRST, SECOND, THIRD = "test/first", "test/second", "test/third"
"""Three ids carrying a `/`, as every Hub id does, and named for the order they are loaded
in — which is the order `/admin/state` lists them back in."""

_DEADLINE = 15.0
"""Every wait is bounded by it: a sweep that never fires or a job that never lands fails
here instead of hanging a suite that has no global timeout."""

_TTL_SECONDS = 1
"""The shortest TTL `/admin/config` accepts (`gt=0`, and an int), which is what makes the
resolution of the sweep observable at all in a test."""

_LATENESS = 1.0
"""How late the sweep may be past a deadline it knew in advance. It is the assertion that
separates waiting for the deadline from polling on an interval: an adaptive polling interval
falls to 10-30s while everything is idle, and nothing else in this test can tell that apart
from an expiry that fires on time. A second is two orders above the sweep's own path — one
sqlite read and one trip through the queue — and one order above the 20 ms the poll below
notices with."""

_WEIGHT_BYTES = 128 * 1024 * 1024

_ROWS = _WEIGHT_BYTES // (4 * 4)

_SLACK = 16 * 1024 * 1024
"""What the measured ceiling leaves for the process to drift by. The two meters are the
process's, not this model's, and between the reading that sets the limit and the load that
has to cross it the suite allocates on its own — a request's json, a sqlite page, the
activations of the generation in between. It has to stay well under `_WEIGHT_BYTES`, or the
load that should cross the ceiling fits under it and nothing is evicted; and well over the
drift, or that load evicts twice. Eight times the drift `test_residency` measured around a
load, and an eighth of the weights it has to stay distinguishable from.
"""


class Weights(nn.Module):
    """One embedding and nothing else: `resident_bytes` over this tree is `_WEIGHT_BYTES`,
    which is also what the header written to disk declares — the accounting and the admission
    figure are about the same model, or the arithmetic above means nothing."""

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


class GatedLanguageModel:
    """One token and then a wait. What makes a request in flight is not a sleep long enough
    to outlast a load — it is the test deciding when the generation ends, so the pressure
    below arrives while the job is provably still the queue's."""

    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        for index in range(options.max_tokens):
            if index == 1:
                assert self.gate.wait(_DEADLINE), "the gate never opened"
            yield Segment("content", str(index))


def gated(gate: threading.Event) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(GatedLanguageModel(gate), [])


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The two caches the scan reads, pointed away from the machine's own: admission sizes
    the incoming model by scanning them, and a test that left them alone would size the
    checkpoints the user actually has on disk."""
    return caches_at(monkeypatch, tmp_path)


def installed(hub: Path, model_id: str, weighs: int = _WEIGHT_BYTES) -> None:
    """The model as the catalog scan finds it: `refs/main` naming a snapshot, a config with a
    `model_type`, and a shard whose header declares the bytes. No payload is written — what
    admission sums is `data_offsets`, and 128 MiB of zeros on disk would buy the suite
    nothing."""
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [weighs // 4], "data_offsets": [0, weighs]}}
    ).encode()
    repository = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "test"}))
    (snapshot / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("head")


@asynccontextmanager
async def serving(loader: Loader) -> AsyncGenerator[tuple[httpx.AsyncClient, Engine]]:
    """The daemon's own app, run through its own lifespan: `ASGITransport` runs none, and the
    lifespan is what migrates the database, opens it, starts the jobs pool and the engine —
    which is also what starts and stops the sweep."""
    daemon = Daemon()
    engine = Engine(loader, daemon, Metrics())
    app = app_of(engine, daemon)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://admission") as http:
            yield http, engine


def ceiling() -> int:
    """The limit that leaves the daemon exactly full: what the two meters admission reads say
    the process is holding now, and room for nothing more.

    Measured rather than written down, and measured with the models already in: what a
    constant would mean depends on which tests ran before this file. It also never has to
    move by a model's weight for the test below to mean anything — the eviction loop takes
    the room it makes off its own accounting, so the only thing this figure has to be is
    stable to within `_SLACK` between here and the next load.
    """
    return max(mx.get_active_memory(), footprint_bytes()) + _SLACK


async def configure(http: httpx.AsyncClient, **values: int) -> None:
    response = await asyncio.wait_for(http.patch("/admin/config", json=values), _DEADLINE)
    assert response.status_code == 200, response.text


async def load(http: httpx.AsyncClient, model_id: str) -> None:
    """The PUT that loads, and the job it hands back run to its end."""
    response = await asyncio.wait_for(http.put(f"/admin/models/{model_id}/residency"), _DEADLINE)
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + _DEADLINE
    while True:
        view = await asyncio.wait_for(http.get(f"/admin/jobs/{job_id}"), _DEADLINE)
        assert view.status_code == 200, view.text
        state = view.json()
        if state["state"] in {"ok", "error", "cancelled"}:
            assert state["state"] == "ok", state["error"]
            return
        assert time.monotonic() < deadline, f"the load stayed {state['state']!r}"
        await asyncio.sleep(0.005)


async def resident(http: httpx.AsyncClient) -> list[dict[str, object]]:
    response = await asyncio.wait_for(http.get("/admin/state"), _DEADLINE)
    assert response.status_code == 200, response.text
    models = response.json()["models"]
    assert isinstance(models, list)
    return models


async def expired(http: httpx.AsyncClient) -> None:
    """Waits for the sweep to empty `/admin/state`. A TTL that never fires fails on the
    deadline instead of leaving the suite waiting on it."""
    deadline = time.monotonic() + _DEADLINE
    while models := await resident(http):
        assert time.monotonic() < deadline, f"still resident past the TTL: {models}"
        await asyncio.sleep(0.02)


async def drain(job: Job) -> list[Segment]:
    """Bounded on purpose: a sentinel that stops being pushed has to fail the test rather
    than hang the suite."""
    pieces: list[Segment] = []
    while (piece := await asyncio.wait_for(job.chunks.get(), _DEADLINE)) is not None:
        pieces.append(piece)
    return pieces
