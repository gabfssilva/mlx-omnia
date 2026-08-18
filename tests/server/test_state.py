"""/admin/state: what is resident, what it costs, and how deep the queue is.

The number this route exists to get right is the residency total, and the way to get it
wrong is to trust a live meter: after a model settles both MLX's active memory and the
process' resident size read below what it occupies, which is how another MLX server once
let a second large model in and blew the ceiling. The test that matters here sets both
meters below the accumulator on purpose — the situation cannot be produced by allocating, only by
simulating the meters that lie.

The two prefix tiers this route also reports are `test_state_prefixes.py`'s.
"""

import asyncio
import time
from importlib import import_module

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia import CompositeModel, GenerationOptions, Text
from mlx_omnia.engine.footprint import resident_bytes
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime import admission as admission_module
from mlx_omnia.server.runtime.engine import Engine, tree
from tests.server.state_stand import (
    BIG_BYTES,
    KV_BYTES,
    SPIKE_BYTES,
    TINY_BYTES,
    HoldingLanguageModel,
    big,
    drain,
    holding,
    models_of,
    piece,
    read,
    tiny,
    within,
)


def test_at_boot_the_list_is_empty_and_the_queue_is_zero() -> None:
    """The memory rail reads this before any request has named a model, and an engine
    that reported itself busy at boot would draw a queue that does not exist."""
    body = within(read(Engine(lambda _: tiny())))
    assert models_of(body) == []
    assert body["queue"] == {"running": 0, "waiting": 0, "reserved": False}


def test_the_bytes_come_off_the_tree_under_the_wrappers() -> None:
    """The engine holds a `LanguageModel`, a protocol over `stream`, and the byte
    arithmetic wants the `nn.Module`. Lose the walk down the wrappers and every resident
    model reports zero weights — which is the rail drawing an empty bar over a loaded 30B.
    """

    async def run() -> dict[str, object]:
        engine = Engine(lambda _: tiny())
        await engine.resolve("tiny")
        return await read(engine)

    body = within(run())
    assert models_of(body)[0]["weights_bytes"] == TINY_BYTES
    assert tree(CompositeModel(HoldingLanguageModel(), [])) is None, "no tree, no tensors"


def test_a_generation_leaves_the_model_with_kv_and_a_recent_last_use() -> None:
    """KV is counted apart from the weights because it grows per request: a limit that
    only knows the weights holds until the first long context and then stops holding."""

    async def run() -> tuple[dict[str, object], float]:
        engine = Engine(lambda _: holding())
        engine.start()
        try:
            # A high-water mark set and released before the request: what is bounded below
            # is *this* request's cost, and without the peak reset the figure would be
            # whatever the process happened to touch since it booted.
            spike = mx.zeros((SPIKE_BYTES // 4,), dtype=mx.float32)
            mx.eval(spike)
            del spike
            mx.clear_cache()
            before = time.time()
            job = await engine.submit("held", Text("hello"), GenerationOptions(max_tokens=4))
            await drain(job)
            return await read(engine), before
        finally:
            await engine.stop()

    body, before = within(run())
    entry = models_of(body)[0]
    assert entry["id"] == "held"
    kv_bytes = entry["kv_bytes"]
    assert isinstance(kv_bytes, int)
    # Bounded above as well as below: without `reset_peak_memory` the figure is the whole
    # session's high-water minus this request's settled memory, which is still over the
    # floor and is not this request's cost.
    assert KV_BYTES <= kv_bytes < 2 * KV_BYTES
    assert body["kv_bytes"] == kv_bytes, "the total is the sum, and there is one model"
    last_used = entry["last_used"]
    assert isinstance(last_used, float)
    assert before <= last_used <= time.time()


def test_a_generation_that_ends_refreshes_the_last_use() -> None:
    """The stamp the arrival left is the *start* of the request. A TTL sweep reading only
    that one evicts a model half an hour after it began a generation that just ended."""

    async def run() -> tuple[object, object]:
        engine = Engine(lambda _: holding(delay=0.01))
        engine.start()
        try:
            job = await engine.submit("held", Text("hi"), GenerationOptions(max_tokens=16))
            assert await piece(job) is not None, "the job never started"
            accepted = models_of(await read(engine))[0]["last_used"]
            await drain(job)
            return accepted, models_of(await read(engine))[0]["last_used"]
        finally:
            await engine.stop()

    accepted, finished = within(run())
    assert isinstance(accepted, float) and isinstance(finished, float)
    assert finished > accepted


def test_a_second_request_moves_the_last_use_and_does_not_duplicate_the_entry() -> None:
    """A model asked for twice is one model. Keying the accounting by anything but the id
    would double its bytes in the total, which is a ceiling reached at half the memory."""
    loads: list[str] = []

    async def run() -> tuple[dict[str, object], dict[str, object]]:
        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return holding()

        engine = Engine(loader)
        engine.start()
        try:
            first = await engine.submit("held", Text("a"), GenerationOptions(max_tokens=2))
            await drain(first)
            before = await read(engine)
            await asyncio.sleep(0.01)
            second = await engine.submit("held", Text("b"), GenerationOptions(max_tokens=2))
            await drain(second)
            return before, await read(engine)
        finally:
            await engine.stop()

    before, after = within(run())
    assert loads == ["held"]
    assert len(models_of(before)) == len(models_of(after)) == 1
    assert models_of(after)[0]["loaded_at"] == models_of(before)[0]["loaded_at"]
    earlier = models_of(before)[0]["last_used"]
    later = models_of(after)[0]["last_used"]
    assert isinstance(earlier, float) and isinstance(later, float)
    assert later > earlier


def test_two_requests_racing_a_cold_load_leave_one_entry_that_keeps_its_stamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two callers naming the same cold model share one load, and both come back out of it.
    Only the first may write the entry: the second overwrites `last_used` with `None` for a
    model a request is already using, and walks the tree of a 30B a second time for bytes
    that are already known.

    What makes it visible is counting the walk: a stamp can be restored by whatever runs
    next, and a second walk over a 30B's ~1500 modules cannot be taken back.
    """
    loads: list[str] = []
    walks: list[object] = []

    def counting(module: nn.Module) -> int:
        walks.append(module)
        return resident_bytes(module)

    monkeypatch.setattr(admission_module, "resident_bytes", counting)

    async def run() -> dict[str, object]:
        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            time.sleep(0.05)
            return tiny()

        engine = Engine(loader)
        engine.start()
        try:
            job, _ = await asyncio.wait_for(
                asyncio.gather(
                    engine.submit("tiny", Text("a"), GenerationOptions(max_tokens=1)),
                    engine.resolve("tiny"),
                ),
                30,
            )
            await drain(job)
            return await read(engine)
        finally:
            await engine.stop()

    entries = models_of(within(run()))
    assert loads == ["tiny"], "the load was not shared"
    assert walks == walks[:1], "the tree was walked once per caller instead of once"
    assert len(entries) == 1
    assert entries[0]["weights_bytes"] == TINY_BYTES
    assert isinstance(entries[0]["last_used"], float)


def test_a_model_that_lands_mid_generation_is_not_charged_to_the_running_one() -> None:
    """Generation is serialized by the gate; loading is deliberately outside it. A model
    that lands while another is decoding puts its whole weight into the same MLX peak, and
    charging that to the running model's KV counts one allocation twice — on a 30B, tens of
    gigabytes attributed to the wrong model, and pinned there until its next request."""

    async def run() -> dict[str, object]:
        engine = Engine(lambda model_id: holding(delay=0.02) if model_id == "held" else big())
        engine.start()
        try:
            job = await engine.submit("held", Text("hi"), GenerationOptions(max_tokens=16))
            assert await piece(job) is not None, "the generation never started"
            await asyncio.wait_for(engine.resolve("big"), 30)
            await drain(job)
            return await read(engine)
        finally:
            await engine.stop()

    entries = {str(entry["id"]): entry for entry in models_of(within(run()))}
    kv_bytes = entries["held"]["kv_bytes"]
    assert isinstance(kv_bytes, int)
    assert kv_bytes < BIG_BYTES, "the loaded model's weights landed in the other model's KV"


def test_the_total_is_the_maximum_and_never_a_live_meter_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A6, and the only way to reproduce it: once a model settles both live meters read
    *below* what it occupies, so they are set below the accumulator here on purpose. Drop
    the accumulator from the maximum and the daemon admits a second large model on a
    reading that already forgot the first — which is how another MLX server once blew
    its ceiling.
    """

    async def run() -> dict[str, object]:
        engine = Engine(lambda _: tiny())
        await engine.resolve("tiny")
        return await read(engine)

    monkeypatch.setattr(mx, "get_active_memory", lambda: 1)
    route = import_module("mlx_omnia.server.api.management.state")
    monkeypatch.setattr(route, "footprint_bytes", lambda: 2)
    assert within(run())["resident_bytes"] == TINY_BYTES

    # And the other side of the maximum: a meter above the accumulator is what gets
    # reported, because it sees what the headers never do — quantize-on-load, the traces.
    monkeypatch.setattr(mx, "get_active_memory", lambda: TINY_BYTES * 10)
    assert within(run())["resident_bytes"] == TINY_BYTES * 10


def test_the_queue_reports_what_runs_and_what_waits() -> None:
    """The depth the app prints, which is literal today. The gate serializes generation,
    so a busy engine is one running and the rest waiting — never two running.

    Read with a request in flight, which is also when a model must not read as idle: the
    stamp that says so is the one the request's arrival leaves, not the one its end does.
    """

    async def run() -> dict[str, object]:
        engine = Engine(lambda _: holding(delay=0.01))
        engine.start()
        try:
            jobs = [
                await engine.submit("held", Text("hello"), GenerationOptions(max_tokens=64))
                for _ in range(3)
            ]
            assert await piece(jobs[0]) is not None, "the first job never started"
            body = await read(engine)
            for job in jobs:
                job.cancel()
            for job in jobs:
                await drain(job)
            return body
        finally:
            await engine.stop()

    body = within(run())
    assert body["queue"] == {"running": 1, "waiting": 2, "reserved": False}
    assert models_of(body)[0]["last_used"] is not None, "a model mid-request reads as idle"
