"""Residency under pressure: the ceiling that evicts, the TTL that expires, and the lease
that neither of them is allowed to take.

No checkpoint is opened, but one is *written*. Admission sizes what it is about to load off
the safetensors headers — the only figure that exists before the load — so the fake cache
here carries a header that declares the bytes with no payload behind it, and the double the
loader hands back is what actually occupies them.

Every limit is set through `PATCH /admin/config`, which is where the daemon reads it, and
every limit is **measured** before it is set: the two meters admission decides against are
the whole process's, so a ceiling written down as a constant would evict everything or
nothing depending on which tests ran before this file.
"""

import asyncio
import gc
import json
import threading
import time
import weakref
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
import pytest

from sideros import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    KVCache,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
)
from sideros.suppress import Segment
from sideros_server import Engine, catalog, create_app
from sideros_server.engine import Job, ModelTooLarge
from sideros_server.state import footprint_bytes
from sideros_server.store import Store

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
separates waiting for the deadline from polling on an interval: oMLX's adaptive interval
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

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.embed(ids)


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [0]

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
    directory = tmp_path / "hub"
    monkeypatch.setattr(catalog, "HUB_CACHE", directory)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    return directory


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
async def serving(engine: Engine, store: Store) -> AsyncGenerator[httpx.AsyncClient]:
    """The daemon's own app, started by hand: `ASGITransport` runs no lifespan, and what the
    lifespan does is the `start`/`stop` around this block — which is also what starts and
    stops the sweep."""
    app = create_app(engine, store)
    engine.start()
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://admission") as http:
            yield http
    finally:
        engine.stop()


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


def test_the_load_that_crosses_the_ceiling_evicts_the_least_recently_used(
    hub: Path, tmp_path: Path
) -> None:
    """The whole of the LRU half: with two models in and the ceiling set where they leave it,
    the third load makes room, and the model it takes the room from is the one nobody has
    asked for in longest.

    The request in the middle is what makes this a test of recency and not of load order —
    with `loaded_at` alone, evicting the oldest entry and evicting the least recently used are
    the same answer, and a `popitem` would pass. And exactly one model goes: a loop that kept
    evicting against a meter that has not caught up yet would empty the daemon.
    """

    async def run() -> list[dict[str, object]]:
        for model_id in (FIRST, SECOND, THIRD):
            installed(hub, model_id)
        store = Store(tmp_path / "server.db")
        engine = Engine(lambda _: weighted(), store)
        async with serving(engine, store) as http:
            await load(http, FIRST)
            await load(http, SECOND)
            await configure(http, memory_limit_bytes=ceiling())
            await drain(await engine.submit(FIRST, Text("hi"), GenerationOptions(max_tokens=1)))
            await load(http, THIRD)
            return await resident(http)

    models = asyncio.run(run())
    assert [model["id"] for model in models] == [FIRST, THIRD]


def test_a_model_larger_than_the_whole_ceiling_is_refused_and_evicts_nothing(
    hub: Path, tmp_path: Path
) -> None:
    """The second invariant. There is no set of evictions that makes room for a checkpoint
    heavier than the whole limit, so the answer is a named error before the first one — a
    loop chasing that space would hand the client the same failure with every other model
    thrown out of memory on the way.

    The checkpoint is never opened either: the size comes off the header, and a refusal that
    still paid for the load would be a refusal after the damage.
    """
    loads: list[str] = []

    async def run() -> tuple[str, list[dict[str, object]]]:
        installed(hub, FIRST)
        installed(hub, SECOND)
        store = Store(tmp_path / "server.db")

        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return weighted()

        engine = Engine(loader, store)
        async with serving(engine, store) as http:
            await load(http, FIRST)
            await configure(http, memory_limit_bytes=_WEIGHT_BYTES // 2)
            with pytest.raises(ModelTooLarge) as refused:
                await engine.resolve(SECOND)
            return str(refused.value), await resident(http)

    message, models = asyncio.run(run())
    assert SECOND in message and str(_WEIGHT_BYTES // 2) in message
    assert loads == [FIRST], "the checkpoint was opened before the size was judged"
    assert [model["id"] for model in models] == [FIRST], "evicted for space that cannot exist"


def test_the_idle_ttl_expires_and_the_weights_actually_come_back(
    hub: Path, tmp_path: Path
) -> None:
    """The reading is MLX's and never the engine's accounting, which reports the model gone
    the instant the key is dropped: the bug this task can introduce is exactly a sweep whose
    bookkeeping and whose memory disagree.

    Two readings, because active memory alone does not tell the whole story. A sweep that
    dropped the dict entry on the loop instead of sending the reference down the queue also
    takes the model out of `get_active_memory()` — and leaves the buffers in MLX's own cache,
    which is inside the process footprint admission decides against. `clear_cache()` empties
    the whole process's cache, so a reading near zero here is about this unload whatever the
    neighbouring tests left behind.

    Both memory ends are deltas rather than a return to `before`: `get_active_memory()` is the
    process's number, so whatever else the suite left alive is under both readings.

    And the reading waits for the model *object* to die rather than taking the meter the
    instant `/admin/state` empties. Between the two there is a reference the engine does not
    hold: the route ran the load through `run_coroutine_threadsafe`, and the chain of futures
    carrying its result is freed on a later turn of the loop. Reading before that turn
    measures asyncio's timing and not the sweep — which is what made this assertion fail
    about one run in four under a random test order, always with the delta at exactly zero.
    """
    alive: list[weakref.ref[object]] = []

    def loader(_: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        model = weighted()
        alive.append(weakref.ref(model))
        return model

    async def run() -> tuple[int, int, int, int, float]:
        installed(hub, FIRST)
        store = Store(tmp_path / "server.db")
        engine = Engine(loader, store)
        async with serving(engine, store) as http:
            await configure(http, idle_ttl_seconds=_TTL_SECONDS)
            before = mx.get_active_memory()
            await load(http, FIRST)
            occupied = mx.get_active_memory()
            loaded = time.monotonic()
            await expired(http)
            waited = time.monotonic() - loaded
            deadline = time.monotonic() + _DEADLINE
            # `collect` takes only unreachable objects, so a reference the engine itself kept
            # survives every turn of this loop and fails on the deadline — which is the leak
            # this test is here for, told apart from the loop's own bookkeeping by waiting.
            while alive[0]() is not None:
                assert time.monotonic() < deadline, "the entry expired and the model did not"
                gc.collect()
                await asyncio.sleep(0.02)
            return before, occupied, mx.get_active_memory(), mx.get_cache_memory(), waited

    before, occupied, freed, cached, waited = asyncio.run(run())
    assert occupied - before >= _WEIGHT_BYTES - _SLACK, "the load never materialized the weights"
    assert occupied - freed >= _WEIGHT_BYTES - _SLACK, "the entry expired and the memory did not"
    assert cached < _WEIGHT_BYTES, "the buffers stayed in MLX's cache instead of going back"
    assert waited < _TTL_SECONDS + _LATENESS, f"the sweep took {waited:.2f}s to notice"


def test_a_queued_request_keeps_its_model_through_the_pressure(hub: Path, tmp_path: Path) -> None:
    """The first invariant, and the reason the mechanism is a lease and not a look at the
    scheduler. Under a ceiling nothing can satisfy, admission wants every resident model:
    both survive, because both are held.

    The one that discriminates is the second. Its job is queued and not running, so an
    eviction that asked the scheduler what it is decoding — the obvious thing to ask, and
    the mutation this test exists for — would read it as idle and take it. The first is
    decoding and would survive that reading too.

    The load goes through over the limit rather than failing: there is nothing left to evict,
    and refusing a request because two others are in flight is the worse answer.
    """
    gate = threading.Event()

    async def run() -> tuple[list[dict[str, object]], list[Segment], list[Segment]]:
        for model_id in (FIRST, SECOND, THIRD):
            installed(hub, model_id)
        store = Store(tmp_path / "server.db")
        engine = Engine(lambda _: gated(gate), store)
        async with serving(engine, store) as http:
            running = await engine.submit(FIRST, Text("hi"), GenerationOptions(max_tokens=4))
            opening = await asyncio.wait_for(running.chunks.get(), _DEADLINE)
            assert opening is not None, "the generation never started"
            queued = await engine.submit(SECOND, Text("hi"), GenerationOptions(max_tokens=4))
            # Under everything the process is holding: every resident model is wanted for
            # eviction, and the incoming one still fits the limit on its own, so the refusal
            # above is not what this test is measuring.
            await configure(http, memory_limit_bytes=_WEIGHT_BYTES)
            await engine.resolve(THIRD)
            models = await resident(http)
            gate.set()
            return models, [opening, *await drain(running)], await drain(queued)

    models, decoding, waiting = asyncio.run(run())
    assert [model["id"] for model in models] == [FIRST, SECOND, THIRD]
    assert decoding == waiting == [Segment("content", str(index)) for index in range(4)]
