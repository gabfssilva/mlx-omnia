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
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

from mlx_omnia import CompositeModel, GenerationOptions, Text
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.services.jobs.registry import _WORKERS
from tests.server.residency_stand import (
    _DEADLINE,
    _SLACK,
    _STORM,
    _WEIGHT_BYTES,
    MODEL,
    caches,
    drain,
    load,
    occupied,
    resident,
    serving,
    settled,
    slow,
    unload,
    url,
    weighted,
)

__all__ = ["caches"]


def test_a_put_loads_the_model_and_the_state_lists_what_it_took() -> None:
    """The screen's Load button: 202 with a job to watch, and once the job ends the model is
    in `/admin/state` weighing what it actually took — an id that lands there weighing nothing
    is the rail drawing an empty bar over a loaded model."""

    async def run() -> tuple[dict[str, object], list[dict[str, object]]]:
        async with serving(lambda _: weighted()) as (http, _engine):
            view = await load(http, MODEL)
            return view, await resident(http)

    view, models = asyncio.run(run())
    assert view["state"] == "ok", view["error"]
    assert [model["id"] for model in models] == [MODEL]
    assert models[0]["weights_bytes"] == _WEIGHT_BYTES


def test_a_put_on_a_resident_model_finishes_without_loading_it_again() -> None:
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

        async with serving(loader) as (http, _engine):
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


def test_a_delete_gives_the_weights_back_and_not_only_the_key() -> None:
    """The whole of the task: dropping the dict entry is not unloading. The reading is MLX's
    and never the engine's accounting, which would report the model gone either way; and the
    buffer cache goes with it, because what an unload is for is a footprint that came down —
    the number admission decides against, and the one a rail can draw."""

    async def run() -> tuple[int, int, int, int, list[dict[str, object]]]:
        async with serving(lambda _: weighted()) as (http, _engine):
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


def test_a_delete_with_a_generation_in_flight_waits_for_it_and_loses_no_token() -> None:
    """The behaviour the task had to pick: the DELETE neither refuses nor interrupts. It takes
    its place in the same queue the gate drains, so the reference is let go between two jobs
    and never under a decode thread that is still reading the weights.

    That the request had already ended when the 204 came back is what `job.state` says: an
    unload that dropped its reference on the loop answers while the job is still `running`,
    and the tokens after it would be generated over memory that was handed back.
    """

    async def run() -> tuple[str, list[Segment]]:
        async with serving(lambda _: slow(0.02)) as (http, engine):
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


def test_a_delete_answers_404_for_a_model_that_is_not_resident() -> None:
    """Both ways of not being resident: never loaded, and unloaded a moment ago. A second 204
    would tell the screen that a button did something it did not do, and nothing on disk is
    consulted for either — the model here was never downloaded.

    The 204 in the middle is also what holds the mounting order in place. The catalog's
    `DELETE /admin/models/{model_id:path}` matches slashes, so registered before this router it
    answers `.../residency` as a model name of its own — and it answers it with a 404, which
    is exactly what the two ends of this test expect to see.
    """

    async def run() -> tuple[int, int, int]:
        async with serving(lambda _: weighted()) as (http, _engine):
            cold = await unload(http, MODEL)
            view = await load(http, MODEL)
            assert view["state"] == "ok", view["error"]
            dropped = await unload(http, MODEL)
            again = await unload(http, MODEL)
            return cold.status_code, dropped.status_code, again.status_code

    assert asyncio.run(run()) == (404, 204, 404)


def test_a_profile_named_residency_still_belongs_to_the_profiles_route() -> None:
    """The other side of the same converter, and the reason this router goes *after*
    `profiles`: `/admin/models/{id:path}/residency` matches `.../m/profiles/residency` with
    the id reading `m/profiles`. Registered first, it would answer a profile's PUT with a load
    job for a model nobody named.
    """

    async def run() -> tuple[int, object]:
        async with serving(lambda _: weighted()) as (http, _engine):
            response = await asyncio.wait_for(
                http.put(f"/admin/models/{MODEL}/profiles/residency", json={}), _DEADLINE
            )
            return response.status_code, response.json()

    status, body = asyncio.run(run())
    assert status == 200, body
    assert isinstance(body, dict)
    assert (body["model"], body["name"]) == (MODEL, "residency")


def test_a_storm_of_loads_finishes_and_a_generation_in_the_middle_still_answers() -> None:
    """The cycle, and the one test that tells the two designs of task 48 apart.

    A load's body blocks its thread on a coroutine, and that coroutine needs a thread of the
    loop's own to finish: `Engine._load` hands the loader to one, and so do admission's
    `_checkpoint_size` and `Engine._generate`. Run the bodies on that same executor and enough
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
        async with serving(loader) as (http, engine):
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


def test_a_load_cancelled_while_it_reads_gives_the_weights_back() -> None:
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
        async with serving(loader) as (http, _engine):
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
