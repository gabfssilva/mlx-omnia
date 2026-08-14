"""/admin/state: what is resident, what it costs, and how deep the queue is.

The number this route exists to get right is the residency total, and the way to get it
wrong is to trust a live meter: after a model settles both MLX's active memory and the
process' resident size read below what it occupies, which is how another MLX server once
let a second large model in and blew the ceiling. The test that matters here sets both
meters below the accumulator on purpose — the situation cannot be produced by allocating, only by
simulating the meters that lie.
"""

import asyncio
import gc
import time
from collections.abc import Iterator
from pathlib import Path
from tempfile import mkdtemp
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi import FastAPI

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
from mlx_omnia.engine.footprint import resident_bytes
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server import engine as engine_module
from mlx_omnia.server import state
from mlx_omnia.server.engine import Engine, Job, tree
from mlx_omnia.server.state import router
from mlx_omnia.server.store import PrefixCacheFile, Store

_KV_BYTES = 32 * 1024 * 1024

_SPIKE_BYTES = 256 * 1024 * 1024
"""Larger than any figure a correct KV reading can produce, and released before the request
that is measured: it is the peak that must not survive into the next request's number."""

TINY_BYTES = 8 * 4 * 4
"""The embedding below and nothing else: 8 rows of 4 float32."""


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.embed(ids)


_WIDE = 16 * 1024
"""Columns per cached row. The KV buffer grows in blocks of 256 rows, so one prompt reserves
`256 * _WIDE * 4` bytes twice over — 32 MiB, which is a figure that stands clear of whatever
the allocator is doing around it."""

_WIDE_TRIE_BYTES = 2 * 256 * _WIDE * 4


class WideLM(nn.Module):
    """`TinyLM` with a cache worth measuring: what the trie keeps between requests is these
    buffers, and at four kilobytes the question of whether anything sees them cannot be
    asked."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        out = self.embed(ids)
        if cache is None:
            return out
        rows = mx.zeros((1, 1, ids.shape[1], _WIDE), dtype=mx.float32)
        keys, values = cache[0].update_and_fetch(rows, rows)
        # The head reads both buffers back, the way a real trunk does, and that is what makes
        # them exist: mlx is lazy, so a fake whose output ignores its own cache leaves the
        # write unevaluated — and memory nothing allocated is memory nothing measures.
        return out + keys[0, 0, -1, 0] + values[0, 0, -1, 0]


def wide() -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(TextLanguageModel(WideLM(), TinyTokenizer()), [])


class TinyTokenizer:
    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        return iter([0])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"."


def tiny() -> CompositeModel[Text, Segment, GenerationOptions]:
    """The production chain down to the tree: the engine holds a `CompositeModel`, which
    holds the task-level model, which holds the `nn.Module` the bytes come off."""
    return CompositeModel(TextLanguageModel(TinyLM(), TinyTokenizer()), [])


class HoldingLanguageModel:
    """Holds an `mx.array` for as long as its stream runs, which is what a KV cache is to
    the accounting: memory the request takes on top of the settled weights and gives back
    when it ends."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        cache = mx.zeros((_KV_BYTES // 4,), dtype=mx.float32)
        mx.eval(cache)
        for index in range(options.max_tokens):
            time.sleep(self.delay)
            yield Segment("content", str(index))


def holding(delay: float = 0.0) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(HoldingLanguageModel(delay), [])


_BIG_BYTES = 128 * 1024 * 1024

_BIG_ROWS = _BIG_BYTES // (4 * 4)


class BigLM(TinyLM):
    """Weights large enough that charging them to another model's KV is unmistakable."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(_BIG_ROWS, 4)


def big() -> CompositeModel[Text, Segment, GenerationOptions]:
    model = BigLM()
    # The load is what allocates: a lazy tree costs nothing, and the point of the test is
    # the memory landing while another model is decoding.
    mx.eval(model.parameters())
    return CompositeModel(TextLanguageModel(model, TinyTokenizer()), [])


_SPARE = Store(Path(mkdtemp()) / "server.db")
"""One database for every reading that does not care about one: the disk figure is a `SUM`
over an empty table, and a file per test would be a temporary directory for a zero."""


def settled() -> int:
    """What mlx holds with nothing in flight: the collector first, because a buffer a dead
    Python object still owns is not the allocator's fault, and `clear_cache` after it, so the
    pool's free blocks are not counted as somebody's."""
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


async def piece(job: Job) -> Segment | None:
    """Bounded on purpose: a sentinel that stops being pushed has to fail the test, not
    hang the suite — there is no global pytest timeout."""
    return await asyncio.wait_for(job.chunks.get(), 30)


async def drain(job: Job) -> None:
    while await piece(job) is not None:
        pass


def mounted(engine: Engine, store: Store | None) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine
    # The route reports what the spilled conversations weigh, which is a row count and lives
    # in the database. In memory for every test that is not about it.
    app.state.store = _SPARE if store is None else store
    app.include_router(router)
    return app


async def read(engine: Engine, store: Store | None = None) -> dict[str, object]:
    transport = httpx.ASGITransport(app=mounted(engine, store))
    async with httpx.AsyncClient(transport=transport, base_url="http://state") as client:
        response = await asyncio.wait_for(client.get("/admin/state"), 30)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


async def clear(engine: Engine, store: Store, tier: str) -> None:
    transport = httpx.ASGITransport(app=mounted(engine, store))
    async with httpx.AsyncClient(transport=transport, base_url="http://state") as client:
        response = await asyncio.wait_for(client.delete(f"/admin/prefixes/{tier}"), 30)
    assert response.status_code == 204, response.text


def models_of(body: dict[str, object]) -> list[dict[str, object]]:
    models = body["models"]
    assert isinstance(models, list)
    for entry in models:
        assert isinstance(entry, dict)
    return models


def test_at_boot_the_list_is_empty_and_the_queue_is_zero() -> None:
    """The memory rail reads this before any request has named a model, and an engine
    that reported itself busy at boot would draw a queue that does not exist."""
    body = asyncio.run(read(Engine(lambda _: tiny())))
    assert models_of(body) == []
    assert body["queue"] == {"running": 0, "waiting": 0}


def test_the_bytes_come_off_the_tree_under_the_wrappers() -> None:
    """The engine holds a `LanguageModel`, a protocol over `stream`, and the byte
    arithmetic wants the `nn.Module`. Lose the walk down the wrappers and every resident
    model reports zero weights — which is the rail drawing an empty bar over a loaded 30B.
    """

    async def run() -> dict[str, object]:
        engine = Engine(lambda _: tiny())
        await engine.resolve("tiny")
        return await read(engine)

    body = asyncio.run(run())
    assert models_of(body)[0]["weights_bytes"] == TINY_BYTES
    assert tree(CompositeModel(HoldingLanguageModel(), [])) is None, "no tree, no tensors"


def test_a_hot_trie_is_live_memory_the_admission_reads(tmp_path: Path) -> None:
    """The prefix trie outlives the request that filled it, so between requests it is live
    allocation and not pool — and `_measured` reads exactly that (`mx.get_active_memory`).
    Confirmed rather than read off the source: it is the figure admission evicts against, and
    a trie invisible to it is a ceiling that lets a second model in on top of it.

    One process, one model, measured before and after the unload that releases it. The trie
    hangs on the model and dies with it (`language.py`), and this model's weights are 128
    bytes, so what the fall between the two readings is made of is the trie.
    """

    async def run() -> tuple[int, int]:
        store = Store(tmp_path / "server.db")
        store.set_config({"prefix_cache_bytes": str(4 * _WIDE_TRIE_BYTES)})
        engine = Engine(lambda _: wide(), store)
        engine.start()
        try:
            await drain(await engine.submit("w", Text("hi"), GenerationOptions(max_tokens=2)))
            held = settled()
            assert await engine.unload("w")
            return held, settled()
        finally:
            engine.stop()

    held, gone = asyncio.run(run())
    assert held - gone >= _WIDE_TRIE_BYTES, "the trie's buffers were never live memory"


def test_the_memory_tier_is_reported_and_clearing_it_hands_it_back(tmp_path: Path) -> None:
    """The two numbers the Server screen's prefix rows read, and the button beside them. The
    trie is filled by a request, because that is the only thing that fills one."""

    async def run() -> tuple[int, int]:
        store = Store(tmp_path / "server.db")
        store.set_config({"prefix_cache_bytes": str(4 * _WIDE_TRIE_BYTES)})
        engine = Engine(lambda _: wide(), store)
        engine.start()
        try:
            await drain(await engine.submit("w", Text("hi"), GenerationOptions(max_tokens=2)))
            held = await read(engine, store)
            await clear(engine, store, "memory")
            emptied = await read(engine, store)
            return _int(held["prefix_memory_bytes"]), _int(emptied["prefix_memory_bytes"])
        finally:
            engine.stop()

    held, emptied = asyncio.run(run())
    assert held >= _WIDE_TRIE_BYTES, "the trie the request filled was never reported"
    assert emptied == 0


def test_clearing_the_disk_tier_takes_the_rows_and_the_files(tmp_path: Path) -> None:
    """The floor that survives a restart, and the one thirty models pile up on. Nothing is
    generated here: what the route has to get right is that the row and the file leave
    together, and `prefixes.forget` is what it leans on to do it."""

    async def run() -> tuple[int, int, bool]:
        store = Store(tmp_path / "disk.db")
        spilled = tmp_path / "w" / "cache.safetensors"
        spilled.parent.mkdir(parents=True)
        spilled.write_bytes(b"x" * 64)
        store.save_prefix_file(
            PrefixCacheFile(
                key="k",
                model="w",
                path=str(spilled),
                ids=b"\x01\x02",
                tokens=2,
                bytes=64,
                created_at=time.time(),
                used_at=time.time(),
            )
        )
        engine = Engine(lambda _: tiny(), store)
        held = await read(engine, store)
        await clear(engine, store, "disk")
        emptied = await read(engine, store)
        return (
            _int(held["prefix_disk_bytes"]),
            _int(emptied["prefix_disk_bytes"]),
            spilled.exists(),
        )

    held, emptied, left = asyncio.run(run())
    assert (held, emptied) == (64, 0)
    assert not left, "the row went and the file stayed"


def _int(value: object) -> int:
    assert isinstance(value, int)
    return value


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
            spike = mx.zeros((_SPIKE_BYTES // 4,), dtype=mx.float32)
            mx.eval(spike)
            del spike
            mx.clear_cache()
            before = time.time()
            job = await engine.submit("held", Text("hello"), GenerationOptions(max_tokens=4))
            await drain(job)
            return await read(engine), before
        finally:
            engine.stop()

    body, before = asyncio.run(run())
    entry = models_of(body)[0]
    assert entry["id"] == "held"
    kv_bytes = entry["kv_bytes"]
    assert isinstance(kv_bytes, int)
    # Bounded above as well as below: without `reset_peak_memory` the figure is the whole
    # session's high-water minus this request's settled memory, which is still over the
    # floor and is not this request's cost.
    assert _KV_BYTES <= kv_bytes < 2 * _KV_BYTES
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
            engine.stop()

    accepted, finished = asyncio.run(run())
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
            engine.stop()

    before, after = asyncio.run(run())
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

    monkeypatch.setattr(engine_module, "resident_bytes", counting)

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
            engine.stop()

    entries = models_of(asyncio.run(run()))
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
            engine.stop()

    entries = {str(entry["id"]): entry for entry in models_of(asyncio.run(run()))}
    kv_bytes = entries["held"]["kv_bytes"]
    assert isinstance(kv_bytes, int)
    assert kv_bytes < _BIG_BYTES, "the loaded model's weights landed in the other model's KV"


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
    monkeypatch.setattr(state, "footprint_bytes", lambda: 2)
    assert asyncio.run(run())["resident_bytes"] == TINY_BYTES

    # And the other side of the maximum: a meter above the accumulator is what gets
    # reported, because it sees what the headers never do — quantize-on-load, the traces.
    monkeypatch.setattr(mx, "get_active_memory", lambda: TINY_BYTES * 10)
    assert asyncio.run(run())["resident_bytes"] == TINY_BYTES * 10


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
            engine.stop()

    body = asyncio.run(run())
    assert body["queue"] == {"running": 1, "waiting": 2}
    assert models_of(body)[0]["last_used"] is not None, "a model mid-request reads as idle"
