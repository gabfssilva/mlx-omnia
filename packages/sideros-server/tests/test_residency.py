"""Residency by hand: the PUT that loads, the DELETE that lets go, and what neither interrupts.

No checkpoint is opened. What these routes have to get right is where the last reference to a
model ends up, and a double holding one measurable `mx.array` is enough to see it: what says
an unload worked is `mx.get_active_memory()` coming back down, never the engine's own
bookkeeping — the accounting reads "not resident" the moment the key is dropped, which is
exactly the failure this task exists to prevent.

The app is `create_app`'s and not a router mounted on a throwaway `FastAPI`: the residency
routes share `/admin/models` with the catalog's and the profiles' `{model_id:path}`, a
converter that matches slashes, so the mounting order is part of what every request here goes
through — and two of the tests below are about nothing else.
"""

import asyncio
import gc
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import ThreadPoolExecutor
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
from sideros.parsers import Segment
from sideros_server import Engine, catalog, create_app
from sideros_server.engine import Job
from sideros_server.jobs import _WORKERS
from sideros_server.store import Store

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
    monkeypatch.setattr(catalog, "HUB_CACHE", tmp_path / "hub")
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")


@asynccontextmanager
async def serving(engine: Engine, tmp_path: Path) -> AsyncGenerator[httpx.AsyncClient]:
    """The daemon's own app, started by hand: `ASGITransport` runs no lifespan, and what the
    lifespan does is the `start`/`stop` around this block."""
    app = create_app(engine, Store(tmp_path / "server.db"))
    engine.start()
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://residency") as http:
            yield http
    finally:
        engine.stop()


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


def test_a_put_loads_the_model_and_the_state_lists_what_it_took(tmp_path: Path) -> None:
    """The screen's Load button: 202 with a job to watch, and once the job ends the model is
    in `/admin/state` weighing what it actually took — an id that lands there weighing nothing
    is the rail drawing an empty bar over a loaded model."""

    async def run() -> tuple[dict[str, object], list[dict[str, object]]]:
        engine = Engine(lambda _: weighted())
        async with serving(engine, tmp_path) as http:
            view = await load(http, MODEL)
            return view, await resident(http)

    view, models = asyncio.run(run())
    assert view["state"] == "ok", view["error"]
    assert [model["id"] for model in models] == [MODEL]
    assert models[0]["weights_bytes"] == _WEIGHT_BYTES


def test_a_put_on_a_resident_model_finishes_without_loading_it_again(tmp_path: Path) -> None:
    """The button pressed twice, or two clients pressing it. The second PUT answers the same
    202 and the load never happens again — counted rather than timed, because what makes the
    test fail for the right reason is that the second load happened, not that it was slow.

    `loaded_at` standing still is the other half: an entry rewritten on top of itself loses
    when the memory was actually taken, which is what a TTL sweep reads.
    """
    loads: list[str] = []

    async def run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return weighted()

        engine = Engine(loader)
        async with serving(engine, tmp_path) as http:
            first = await load(http, MODEL)
            assert first["state"] == "ok", first["error"]
            before = await resident(http)
            second = await load(http, MODEL)
            assert second["state"] == "ok", second["error"]
            return before, await resident(http)

    before, after = asyncio.run(run())
    assert loads == [MODEL], "the second PUT loaded the model again"
    assert len(before) == len(after) == 1
    assert after[0]["loaded_at"] == before[0]["loaded_at"], "the entry was written twice"


def test_a_delete_gives_the_weights_back_and_not_only_the_key(tmp_path: Path) -> None:
    """The whole of the task: dropping the dict entry is not unloading. The reading is MLX's
    and never the engine's accounting, which would report the model gone either way; and the
    buffer cache goes with it, because what an unload is for is a footprint that came down —
    the number admission decides against, and the one a rail can draw."""

    async def run() -> tuple[int, int, int, int, list[dict[str, object]]]:
        engine = Engine(lambda _: weighted())
        async with serving(engine, tmp_path) as http:
            before = mx.get_active_memory()
            view = await load(http, MODEL)
            assert view["state"] == "ok", view["error"]
            occupied = mx.get_active_memory()
            dropped = await unload(http, MODEL)
            assert dropped.status_code == 204, dropped.text
            assert not dropped.content, "204 carries no body"
            # The cache before the collection below and the active memory after it, and the
            # order is the whole of it. `mx.clear_cache()` empties the process's cache, so the
            # instant the DELETE returns is the only one where this reading is about the
            # unload; collecting first would refill it with whatever the rest of the suite
            # left unreachable and measure that instead.
            cached = mx.get_cache_memory()
            # Only unreachable objects, so a reference the engine still holds survives this
            # and still fails below: what it removes is *when* the last one dies. The load's
            # own `asyncio.Task` reaches the collector through a cycle, and whether a gen-0
            # pass had already run by here depends on how much garbage the rest of the suite
            # made — which is a property of the neighbours, not of the unload.
            gc.collect()
            freed = mx.get_active_memory()
            return before, occupied, freed, cached, await resident(http)

    before, occupied, freed, cached, models = asyncio.run(run())
    # Both ends are deltas rather than a return to `before`: the load also materializes MLX's
    # global PRNG key, and a few bytes that never come back would fail this for a reason that
    # has nothing to do with the model.
    assert occupied - before >= _WEIGHT_BYTES - _SLACK, "the load never materialized the weights"
    assert occupied - freed >= _WEIGHT_BYTES - _SLACK, "the weights outlived the unload"
    assert cached < _WEIGHT_BYTES, "the buffers stayed in MLX's cache instead of going back"
    assert models == []


def test_a_delete_with_a_generation_in_flight_waits_for_it_and_loses_no_token(
    tmp_path: Path,
) -> None:
    """The behaviour the task had to pick: the DELETE neither refuses nor interrupts. It takes
    its place in the same queue the gate drains, so the reference is let go between two jobs
    and never under a decode thread that is still reading the weights.

    That the request had already ended when the 204 came back is what `job.state` says: an
    unload that dropped its reference on the loop answers while the job is still `running`,
    and the tokens after it would be generated over memory that was handed back.
    """

    async def run() -> tuple[str, list[Segment]]:
        engine = Engine(lambda _: slow(0.02))
        async with serving(engine, tmp_path) as http:
            job = await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=8))
            first = await asyncio.wait_for(job.chunks.get(), _DEADLINE)
            assert first is not None, "the generation never started"
            dropped = await unload(http, MODEL)
            assert dropped.status_code == 204, dropped.text
            state = job.state
            rest = await drain(job)
            return state, [first, *rest]

    state, tokens = asyncio.run(run())
    assert state == "completed", "the 204 came back while the generation was still running"
    assert tokens == [Segment("content", str(index)) for index in range(8)]


def test_a_delete_answers_404_for_a_model_that_is_not_resident(tmp_path: Path) -> None:
    """Both ways of not being resident: never loaded, and unloaded a moment ago. A second 204
    would tell the screen that a button did something it did not do, and nothing on disk is
    consulted for either — the model here was never downloaded.

    The 204 in the middle is also what holds the mounting order in place. The catalog's
    `DELETE /admin/models/{model_id:path}` matches slashes, so registered before this router it
    answers `.../residency` as a model name of its own — and it answers it with a 404, which
    is exactly what the two ends of this test expect to see.
    """

    async def run() -> tuple[int, int, int]:
        engine = Engine(lambda _: weighted())
        async with serving(engine, tmp_path) as http:
            cold = await unload(http, MODEL)
            view = await load(http, MODEL)
            assert view["state"] == "ok", view["error"]
            dropped = await unload(http, MODEL)
            again = await unload(http, MODEL)
            return cold.status_code, dropped.status_code, again.status_code

    assert asyncio.run(run()) == (404, 204, 404)


def test_a_profile_named_residency_still_belongs_to_the_profiles_route(tmp_path: Path) -> None:
    """The other side of the same converter, and the reason this router goes *after*
    `profiles`: `/admin/models/{id:path}/residency` matches `.../m/profiles/residency` with
    the id reading `m/profiles`. Registered first, it would answer a profile's PUT with a load
    job for a model nobody named.
    """

    async def run() -> tuple[int, object]:
        engine = Engine(lambda _: weighted())
        async with serving(engine, tmp_path) as http:
            response = await asyncio.wait_for(
                http.put(f"/admin/models/{MODEL}/profiles/residency", json={}), _DEADLINE
            )
            return response.status_code, response.json()

    status, body = asyncio.run(run())
    assert status == 200, body
    assert isinstance(body, dict)
    assert (body["model"], body["name"]) == (MODEL, "residency")


def test_a_storm_of_loads_finishes_and_a_generation_in_the_middle_still_answers(
    tmp_path: Path,
) -> None:
    """The cycle, and the one test that tells the two designs of task 48 apart.

    A load's body blocks its thread on a coroutine, and that coroutine needs a thread of the
    loop's own to finish: `Engine._load` hands the loader to one, and so do admission's
    `_checkpoint_size` and `Engine._decode`. Run the bodies on that same executor and enough
    concurrent loads hold every thread the coroutines are waiting for — no load finishes, no
    generation starts, and the interpreter joins the blocked threads on the way out.

    The loop's default pool is shrunk here to what `jobs._WORKERS` is, so `_STORM` fills it
    the way `min(32, cpu + 4)` bodies would on the machine and the assertion that fails under
    a shared executor is the generation's, not the storm's. The loader is gated rather than
    slow: the window stays open until this test closes it, so nothing depends on a sleep being
    long enough. And the generation runs on a model that is already resident, which is what
    separates the two halves — it needs no load, only a thread to decode on.
    """
    gate = threading.Event()

    def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        if model_id != MODEL:
            assert gate.wait(_DEADLINE), "the test never let the storm through"
        return slow(0.0)

    async def run() -> tuple[list[Segment], list[object], list[object]]:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=_WORKERS))
        engine = Engine(loader)
        async with serving(engine, tmp_path) as http:
            first = await load(http, MODEL)
            assert first["state"] == "ok", first["error"]
            started = await asyncio.gather(
                *(asyncio.wait_for(http.put(url(model_id)), _DEADLINE) for model_id in _STORM)
            )
            job_ids = [response.json()["id"] for response in started]
            try:
                await occupied(http, job_ids)
                generation = await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=4))
                try:
                    tokens = await drain(generation)
                except TimeoutError:
                    raise AssertionError(
                        "no token while the loads were in flight: the decode thread is a job's"
                    ) from None
            finally:
                gate.set()
            states = [(await settled(http, job_id))["state"] for job_id in job_ids]
            return tokens, states, [model["id"] for model in await resident(http)]

    tokens, states, models = asyncio.run(run())
    assert tokens == [Segment("content", str(index)) for index in range(4)], (
        "the generation waited on the loads"
    )
    assert states == ["ok"] * len(_STORM), states
    assert set(models) == {MODEL, *_STORM}


def test_a_load_cancelled_while_it_reads_gives_the_weights_back(tmp_path: Path) -> None:
    """An MLX load answers no flag from outside, so a cancel cannot stop it — and the job
    would end labelled `cancelled` with the model resident, the label saying the opposite of
    what `/admin/state` says. What cancelling a load means is "do not keep this".

    The cancel is sent while the loader is provably inside the read: the double waits on a
    gate the test opens, which is what makes this about the window and not about a sleep.
    """
    reading = threading.Event()
    proceed = threading.Event()

    def loader(_: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        reading.set()
        assert proceed.wait(_DEADLINE), "the test never let the load finish"
        return weighted()

    async def run() -> tuple[dict[str, object], list[dict[str, object]]]:
        engine = Engine(loader)
        async with serving(engine, tmp_path) as http:
            started = await asyncio.wait_for(http.put(url(MODEL)), _DEADLINE)
            assert started.status_code == 202, started.text
            job_id = started.json()["id"]
            await asyncio.to_thread(reading.wait, _DEADLINE)
            dropped = await asyncio.wait_for(http.delete(f"/admin/jobs/{job_id}"), _DEADLINE)
            assert dropped.status_code == 202, dropped.text
            proceed.set()
            return await settled(http, job_id), await resident(http)

    view, models = asyncio.run(run())
    assert view["state"] == "cancelled"
    assert models == [], "the job says cancelled and the model stayed loaded"
