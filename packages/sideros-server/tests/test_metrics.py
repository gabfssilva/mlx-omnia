"""/admin/metrics: the register of what each request cost, and the stream that publishes it.

Two of the three properties here are about a request that is *still running* — a client that
connects in the middle is told where it is, and a client that leaves does not take the
generation with it — so the model under the engine is paced: one step every `_PACE`, slow
enough for the test to connect in the middle and for a stall to be visible in the rate the
meter reports.

The server is a real one, like the jobs suite: `TestClient` and `ASGITransport` run the whole
response before handing it over, and a stream that only ends when the client goes away would
never hand anything over at all.

The arithmetic of the aggregates is checked off the register directly, with meters whose
marks are written by hand: over HTTP the rates would be whatever the machine did, and the
question there is whether a four-token request and a hundred-token one weigh the same.
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
import pytest
import uvicorn
from fastapi import FastAPI

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
from sideros.generate import Meter
from sideros.suppress import Segment
from sideros_server import metrics as metrics_module
from sideros_server.engine import Engine
from sideros_server.metrics import Metrics, router

_VOCAB = 8
_WIDTH = 4

TINY_ACTIVE_BYTES = _VOCAB * _WIDTH * 4
"""The embedding and nothing else: 8 rows of 4 float32. There is no `lm_head` in the tree, so
the table is the head too and a decode step reads all of it."""

TINY_CEILING = 490.0 * 1e9 / TINY_ACTIVE_BYTES
"""The tok/s the bandwidth allows for a model that size. Written out here rather than imported
so the test does not divide by whatever `footprint` divides by."""

BIG_ACTIVE_BYTES = 1_711_000_000
"""Qwen3-30B-A3B 4-bit's measured active bytes per token (CLAUDE.md), so the percentage the
aggregate test asserts is one someone can check against the number the house publishes."""

_PACE = 0.02
"""Seconds per decode step of the paced model."""

_PACED_TOKENS = 100
"""Two seconds of generation at `_PACE`: room to connect in the middle and to leave again."""

_TICK = 0.001
"""The step of the model with no tree. Small, but not zero: a decode that took no measurable
time would have no rate to report, and this suite would be asserting on `None`."""


class TinyLM(nn.Module):
    """Logits are the embedding lookup: four of them, all reachable ids, so greedy decoding
    runs for exactly as many steps as it is asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _WIDTH)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.embed(ids)


class PacedLM(TinyLM):
    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        time.sleep(_PACE)
        return super().__call__(ids, cache)


class CountingTokenizer:
    """One id per character, so two prompts of different lengths are two different prompt
    counts — which is the difference the register has to keep."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"."


class OpaqueLanguageModel:
    """A `LanguageModel` and nothing else: no `nn.Module` under it for the byte arithmetic to
    read. It counts through the meter the way `stream_ids` does — one mark per id emitted."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value))
        for index in range(options.max_tokens):
            time.sleep(_TICK)
            meter.token()
            yield Segment("content", str(index))


def load(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
    match model_id:
        case "quick":
            return CompositeModel(TextLanguageModel(TinyLM(), CountingTokenizer()), [])
        case "paced":
            return CompositeModel(TextLanguageModel(PacedLM(), CountingTokenizer()), [])
        case "opaque":
            return CompositeModel(OpaqueLanguageModel(), [])
        case other:
            raise ValueError(f"no model {other!r} in this stand")


@dataclass
class Stand:
    base_url: str
    metrics: Metrics


@pytest.fixture(scope="module")
def stand() -> Iterator[Stand]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine = Engine(load)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    async def generate(model: str, prompt: str, max_tokens: int) -> dict[str, object]:
        """The daemon's own path in miniature — submit, drain, answer — because `app.py`'s
        chat route is another agent's file this wave and the register is fed by the engine,
        not by the dialect."""
        job = await engine.submit(model, Text(prompt), GenerationOptions(max_tokens=max_tokens))
        while await asyncio.wait_for(job.chunks.get(), 60) is not None:
            pass
        return {"state": job.state, "completion_tokens": job.meter.completion_tokens}

    app = FastAPI(lifespan=lifespan)
    app.state.metrics = engine.metrics
    app.include_router(router)
    app.post("/generate")(generate)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield Stand(base_url=f"http://127.0.0.1:{port}", metrics=engine.metrics)
    server.should_exit = True
    thread.join(timeout=5)
    assert not thread.is_alive(), "the stand's server did not shut down"


def entry(value: object) -> dict[str, object]:
    assert isinstance(value, dict), f"expected an object, got {value!r}"
    return value


def entries(body: dict[str, object], key: str) -> list[dict[str, object]]:
    values = body[key]
    assert isinstance(values, list), f"expected a list at {key!r}, got {values!r}"
    return [entry(value) for value in values]


def number(record: dict[str, object], key: str) -> float:
    value = record[key]
    assert isinstance(value, int | float), f"expected a number at {key!r}, got {value!r}"
    return value


def run(stand: Stand, model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    response = httpx.post(
        f"{stand.base_url}/generate",
        params={"model": model, "prompt": prompt, "max_tokens": max_tokens},
        timeout=60,
    )
    assert response.status_code == 200, response.text
    return entry(response.json())


def snapshot(stand: Stand) -> dict[str, object]:
    response = httpx.get(f"{stand.base_url}/admin/metrics", timeout=30)
    assert response.status_code == 200, response.text
    return entry(response.json())


def totals(stand: Stand, model: str) -> tuple[float, float, float]:
    """Requests, prompt tokens and completion tokens on one model. Zeros when the model has
    none yet, so the tests read the difference their own requests made — the register is
    module-scoped and counts since the server started."""
    for record in entries(snapshot(stand), "models"):
        if record["model"] == model:
            return (
                number(record, "requests"),
                number(record, "prompt_tokens"),
                number(record, "completion_tokens"),
            )
    return (0.0, 0.0, 0.0)


def frames(response: httpx.Response, seconds: float = 30.0) -> Iterator[dict[str, object]]:
    """Bounded by the clock, not by the transport: this stream is held open by keep-alives and
    never ends on its own, so a fanout that regressed would hang the suite instead of failing
    the test that reads it."""
    deadline = time.monotonic() + seconds
    for line in response.iter_lines():
        assert time.monotonic() < deadline, "the stream never said what the test waits for"
        if line.startswith("data: "):
            yield entry(json.loads(line.removeprefix("data: ")))


def wait_for_a_token(stand: Stand, seconds: float = 30.0) -> dict[str, object]:
    """Polls until the running request has emitted a token, so what follows happens in the
    middle of a generation and not before it. Bounded: a live entry that never appears fails
    here instead of hanging."""
    deadline = time.monotonic() + seconds
    while True:
        live = snapshot(stand)["live"]
        if isinstance(live, dict) and number(entry(live), "completion_tokens") >= 1:
            return entry(live)
        assert time.monotonic() < deadline, "no running request ever reported a token"
        time.sleep(0.01)


def test_two_sequential_requests_are_two_records_with_their_own_numbers(stand: Stand) -> None:
    """The register is a register and not a last-value: the second request must not overwrite
    the first, and each record must carry what *that* request did — same model, different
    prompt, different length. Newest first, which is the order a dashboard reads."""
    before = totals(stand, "quick")

    run(stand, "quick", "abc", 4)
    run(stand, "quick", "abcdefg", 9)

    recent = entries(snapshot(stand), "requests")[:2]
    assert [record["model"] for record in recent] == ["quick", "quick"]
    assert [record["prompt_tokens"] for record in recent] == [7, 3]
    assert [record["completion_tokens"] for record in recent] == [9, 4]
    assert [record["state"] for record in recent] == ["completed", "completed"]
    for record in recent:
        assert number(record, "ttft") > 0
        rate = number(record, "tokens_per_second")
        assert rate > 0
        assert record["bytes_per_token"] == TINY_ACTIVE_BYTES
        assert number(record, "ceiling_fraction") == pytest.approx(rate / TINY_CEILING)

    after = totals(stand, "quick")
    assert (after[0] - before[0], after[1] - before[1], after[2] - before[2]) == (2, 10, 13)


def test_a_model_with_no_tree_reports_no_bytes_and_no_ceiling(stand: Stand) -> None:
    """`footprint` reads tensors and a model the walk cannot reach has none. The field is
    empty rather than zero: a percentage of a ceiling nobody computed is a number the
    dashboard would print as if it meant something."""
    run(stand, "opaque", "hi", 3)

    record = entries(snapshot(stand), "requests")[0]

    assert record["model"] == "opaque"
    assert record["completion_tokens"] == 3
    assert record["bytes_per_token"] is None
    assert record["ceiling_fraction"] is None
    assert number(record, "tokens_per_second") > 0, "the rate is the meter's, not the tree's"


def test_a_client_that_subscribes_mid_generation_is_told_where_the_request_is(
    stand: Stand,
) -> None:
    """The whole reason the stream opens with the current state. A first frame that carried
    only the next transition would tell a dashboard connecting during a long generation
    nothing at all until it ended — which is when its numbers stop being live.

    The clock is part of the assertion: the silent tick resamples every 0.5s, so a stream
    that answered from the tick instead of from the subscription would still show a live
    request — half a second late, and only because something else was already moving."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(run, stand, "paced", "hello", _PACED_TOKENS)
        started = wait_for_a_token(stand)
        asked = time.monotonic()
        with (
            httpx.Client() as http,
            http.stream("GET", f"{stand.base_url}/admin/metrics/events", timeout=30) as response,
        ):
            assert response.status_code == 200
            first = next(frames(response))
            waited = time.monotonic() - asked
        finished = running.result(timeout=60)

    assert waited < metrics_module._KEEP_ALIVE_SECONDS / 2, (
        "the first frame waited for a tick instead of carrying the state"
    )
    live = first["live"]
    assert isinstance(live, dict), "the first frame carried no running request"
    record = entry(live)
    assert record["model"] == "paced"
    assert record["state"] == "running"
    assert record["bytes_per_token"] == TINY_ACTIVE_BYTES
    assert number(record, "ttft") > 0
    tokens = number(record, "completion_tokens")
    assert number(started, "completion_tokens") <= tokens < _PACED_TOKENS
    assert finished["completion_tokens"] == _PACED_TOKENS


_BURST = 3
"""Requests fired back to back, well inside one keep-alive tick. Each moves the register
twice — one starts, one ends — so the burst owes the stream more frames than resampling
could account for."""


def counted(frame: dict[str, object], model: str) -> float:
    for record in entries(frame, "models"):
        if record["model"] == model:
            return number(record, "requests")
    return 0.0


def test_the_stream_keeps_publishing_after_the_frame_the_subscription_owes(
    stand: Stand,
) -> None:
    """Every other test here reads one frame and stops, which leaves the loop *after* the
    subscription untested: replacing its body with a bare keep-alive, or dropping the fanout
    `_publish` writes to the watchers, keeps the suite green over a GET wearing an SSE costume.

    The clock is what separates publishing from resampling. Once the burst is over nothing
    moves again, so a stream that only resampled on the silent tick would find an unchanged
    snapshot and emit keep-alives for ever; under a budget of `_KEEP_ALIVE_SECONDS` per pair of
    frames it cannot deliver them even while the burst is still running.
    """
    budget = _BURST * metrics_module._KEEP_ALIVE_SECONDS
    with (
        httpx.Client() as http,
        http.stream("GET", f"{stand.base_url}/admin/metrics/events", timeout=30) as response,
    ):
        assert response.status_code == 200
        stream = frames(response, budget)
        before = counted(next(stream), "quick")
        for _ in range(_BURST):
            run(stand, "quick", "hello", 2)
        published = [next(stream) for _ in range(_BURST * 2)]

    assert counted(published[-1], "quick") > before, "every frame carried the opening snapshot"


def test_closing_the_stream_neither_cancels_nor_slows_the_generation(stand: Stand) -> None:
    """The chat's own SSE ends in `job.cancel()` — a client that walks away stops paying for
    tokens. This one must not: watching a generation is not taking part in it, and a `finally`
    that cancelled here would let one dashboard truncate another client's answer.

    The rate is the evidence for the second half, and only the floor of it: a decode thread
    that had to wait on the closed stream falls far under `1/_PACE`, while exceeding it is
    impossible by construction — the paced model sleeps `_PACE` per step, so a ceiling here
    would be an assertion about the double and not about the register.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(run, stand, "paced", "hello", _PACED_TOKENS)
        wait_for_a_token(stand)
        with (
            httpx.Client() as http,
            http.stream("GET", f"{stand.base_url}/admin/metrics/events", timeout=30) as response,
        ):
            assert isinstance(next(frames(response))["live"], dict)
        finished = running.result(timeout=60)

    assert finished["state"] == "completed"
    assert finished["completion_tokens"] == _PACED_TOKENS
    record = entries(snapshot(stand), "requests")[0]
    assert record["state"] == "completed"
    assert record["completion_tokens"] == _PACED_TOKENS
    assert number(record, "tokens_per_second") > 0.25 / _PACE

    deadline = time.monotonic() + 10
    while stand.metrics.watchers:
        assert time.monotonic() < deadline, "the closed stream left its queue subscribed"
        time.sleep(0.02)


def test_the_aggregate_weighs_each_request_by_what_it_generated() -> None:
    """Two requests on one model: two tokens in a second, and a hundred in four. The rate of
    the pair is 102/5 = 20.4 tok/s — the ratio of the totals — and not 13.5, the mean of the
    two rates, which would let a four-token request weigh as much as a hundred-token one.

    ttft is the other way round: one prefill per request, whatever its length, so it averages.
    """
    register = Metrics()
    short = Meter(
        prompt_tokens=5, completion_tokens=3, prefill_started=0.0, first_token=0.5, last_token=1.5
    )
    long = Meter(
        prompt_tokens=7, completion_tokens=101, prefill_started=0.0, first_token=1.5, last_token=5.5
    )

    for meter in (short, long):
        register.begin("big", meter, BIG_ACTIVE_BYTES)
        register.end("completed")

    aggregate = register.snapshot().models[0]
    assert aggregate.model == "big"
    counts = (aggregate.requests, aggregate.prompt_tokens, aggregate.completion_tokens)
    assert counts == (2, 12, 104)
    assert aggregate.tokens_per_second == pytest.approx(20.4)
    assert aggregate.ttft == pytest.approx(1.0)
    assert aggregate.bytes_per_token == BIG_ACTIVE_BYTES
    # 490 GB/s over 1.711 GB/token is 286.4 tok/s, and 20.4 of those is 7.12%.
    assert aggregate.ceiling_fraction == pytest.approx(0.07123, rel=1e-3)


def test_the_ring_forgets_old_requests_and_the_totals_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What is bounded is the detail, never the count: the ring is the window a dashboard
    draws and the aggregates are counters since boot. A total kept by summing the ring would
    start going *down* on the request that pushed the first one off the end."""
    monkeypatch.setattr(metrics_module, "_HISTORY", 2)
    register = Metrics()

    for tokens in (1, 2, 3):
        register.begin("m", Meter(prompt_tokens=1, completion_tokens=tokens), None)
        register.end("completed")

    current = register.snapshot()
    assert [record.completion_tokens for record in current.requests] == [3, 2]
    assert current.models[0].requests == 3
    assert current.models[0].completion_tokens == 6
    assert current.models[0].bytes_per_token is None
    assert current.models[0].ceiling_fraction is None


def test_a_request_that_is_running_is_the_live_entry_and_then_a_record() -> None:
    """`live` is the one field a snapshot cannot answer from history, and it is what the
    stream is watched for. It holds the meter itself — a copy taken at the start would freeze
    the numbers at zero — and it is gone once the request ends, so a finished generation never
    reads as still running."""
    register = Metrics()
    meter = Meter()

    register.begin("m", meter, None)
    meter.prefill(4)
    meter.token()
    live = register.snapshot().live
    assert live is not None
    assert (live.state, live.prompt_tokens, live.completion_tokens) == ("running", 4, 1)

    register.end("cancelled")
    ended = register.snapshot()
    assert ended.live is None
    assert ended.requests[0].state == "cancelled"
    assert ended.requests[0].completion_tokens == 1
