"""The stand the /admin/metrics suites run against: a real server, a paced model, and the
readers that turn its JSON into numbers.

Two of the properties here are about a request that is *still running* — a client that
connects in the middle is told where it is, and a client that leaves does not take the
generation with it — so the model under the engine is paced: one step every `PACE`, slow
enough for a test to connect in the middle and for a stall to be visible in the rate the
meter reports.

The server is a real one, like the jobs suite: `TestClient` and `ASGITransport` run the whole
response before handing it over, and a stream that only ends when the client goes away would
never hand anything over at all.
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
import pytest
import uvicorn
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
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.footprint import SUSTAINED_GBS
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.api.management.state import router
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine

_VOCAB = 8
_WIDTH = 4

TINY_ACTIVE_BYTES = _VOCAB * _WIDTH * 4
"""The embedding and nothing else: 8 rows of 4 float32. There is no `lm_head` in the tree, so
the table is the head too and a decode step reads all of it."""

TINY_CEILING = SUSTAINED_GBS * 1e9 / TINY_ACTIVE_BYTES
"""The tok/s the bandwidth allows for a model that size. The bandwidth is imported and the
arithmetic is not: what the test pins is the ratio the register publishes, and a house that
remeasures its ceiling should not have to remember this file."""

BIG_ACTIVE_BYTES = 1_711_000_000
"""Qwen3-30B-A3B 4-bit's measured active bytes per token (CLAUDE.md), so the percentage the
aggregate test asserts is one someone can check against the number the house publishes."""

PACE = 0.02
"""Seconds per decode step of the paced model."""

PACED_TOKENS = 100
"""Two seconds of generation at `PACE`: room to connect in the middle and to leave again."""

COLD_LOAD = 0.2
"""Seconds the `cold` model takes to load. Long enough that a load reported as the wait it
was cannot be confused with the scheduling around it."""

TICK = 0.001
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

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.embed(ids)


class PacedLM(TinyLM):
    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        time.sleep(PACE)
        return super().__call__(ids, cache)


class CountingTokenizer:
    """One id per character, so two prompts of different lengths are two different prompt
    counts — which is the difference the register has to keep."""

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([0] * len(whole))

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
        text = input.value if isinstance(input.value, str) else "".join(input.value)
        meter.prefill(len(text))
        for index in range(options.max_tokens):
            time.sleep(TICK)
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
        case "cold":
            time.sleep(COLD_LOAD)
            return CompositeModel(TextLanguageModel(TinyLM(), CountingTokenizer()), [])
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
    register = Metrics()
    engine = Engine(load, None, register)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        await engine.stop()

    async def generate(model: str, prompt: str, max_tokens: int) -> dict[str, object]:
        """The daemon's own path in miniature — submit, drain, answer — because `app.py`'s
        chat route is another agent's file this wave and the register is fed by the engine,
        not by the dialect."""
        job = await engine.submit(model, Text(prompt), GenerationOptions(max_tokens=max_tokens))
        while await asyncio.wait_for(job.chunks.get(), 60) is not None:
            pass
        return {"state": job.state, "completion_tokens": job.meter.completion_tokens}

    app = FastAPI(lifespan=lifespan)
    app.state.metrics = register
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
    yield Stand(base_url=f"http://127.0.0.1:{port}", metrics=register)
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
        assert isinstance(live, list)
        for record in live:
            if number(entry(record), "completion_tokens") >= 1:
                return entry(record)
        assert time.monotonic() < deadline, "no running request ever reported a token"
        time.sleep(0.01)
