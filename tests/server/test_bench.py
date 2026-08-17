"""/admin/benches: the daemon's own bench, and the caveat that travels with its numbers.

Two of the properties here are about arithmetic over rounds — the median, and the warmup
that is not one of them — so the model under the engine writes the meter's marks itself
rather than being clocked: a median asserted against `time.perf_counter()` is a median
asserted against whatever the machine was doing while the suite ran. `Meter`'s fields are
exactly what `stream_ids` fills in, a prompt count and three marks, so a double that sets
them is a generation whose numbers are known and not a mechanism the engine lacks.

Two others cannot use it: the percentage of the ceiling divides by bytes summed off an
`nn.Module`, so it runs against a real tree, and a `DELETE` that has to land inside a round
needs a round with duration, so it runs against a paced one.

The app is assembled here rather than through `create_app`: the router's wiring is the
orchestrator's, and this suite has to be able to run before it lands.
"""

import threading
import time
from collections.abc import AsyncGenerator, Generator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TypeIs

import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    KVCache,
    Model,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
)
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server import bench, jobs
from mlx_omnia.server.engine import Engine, Loader
from mlx_omnia.server.store import Store
from tests.server.polling import wait_for

_DEADLINE = 30.0
"""Every wait here is bounded by it: a job that never arrives fails this suite instead of
hanging it — there is no global pytest timeout."""

_VOCAB = 8
_WIDTH = 4

TINY_ACTIVE_BYTES = _VOCAB * _WIDTH * 4
"""The embedding and nothing else: 8 rows of 4 float32. There is no `lm_head` in the tree,
so the table is the head too and a decode step reads all of it."""

TINY_CEILING = 490.0 * 1e9 / TINY_ACTIVE_BYTES
"""The tok/s this machine's bandwidth allows for a model that size, written out rather than
imported so the test does not divide by whatever `footprint` divides by."""

_PACE = 0.01
"""Seconds per decode step of the paced model."""

_LONG = 200
"""Tokens per round of the paced model: two seconds, which is the room a `DELETE` has to
land inside a round rather than after it."""


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


class CountingTokenizer:
    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([0] * len(whole))

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"."


@dataclass
class Scripted:
    """A generation whose numbers are chosen instead of clocked. The marks written here are
    the ones `stream_ids` writes, and `runs` walking off the end of the script is how a run
    nobody asked for — a warmup that was counted, a sixth round — fails the job it is in."""

    script: list[tuple[float, float]]
    """(tok/s, ttft in seconds) per generation, the first being the warmup's."""

    runs: int = 0

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rate, ttft = self.script[self.runs]
        self.runs += 1
        meter.prompt_tokens = len(input.value)
        meter.completion_tokens = options.max_tokens
        meter.prefill_started = 0.0
        meter.first_token = ttft
        meter.last_token = ttft + (options.max_tokens - 1) / rate
        for index in range(options.max_tokens):
            yield Segment("content", str(index))


@dataclass
class Paced:
    """One token every `_PACE`, counted: a round long enough for a `DELETE` to land inside
    it, and a count that says whether it did."""

    started: threading.Event = field(default_factory=threading.Event)
    tokens: int = 0

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
            time.sleep(_PACE)
            self.tokens += 1
            meter.token()
            self.started.set()
            yield Segment("content", str(index))


def composite(
    model: Model[Text, Segment, GenerationOptions],
) -> CompositeModel[Text, Segment, GenerationOptions]:
    """The production chain down to the tree: the engine holds a `CompositeModel`, and what
    the byte arithmetic walks for is the `nn.Module` under it — which the two doubles above
    do not have, and `tiny` does."""
    return CompositeModel(model, [])


def tiny() -> CompositeModel[Text, Segment, GenerationOptions]:
    return composite(TextLanguageModel(TinyLM(), CountingTokenizer()))


@contextmanager
def stand(loader: Loader, path: Path) -> Generator[TestClient]:
    engine = Engine(loader)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    store = Store(path)
    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.state.jobs = jobs.Jobs(store)
    app.include_router(bench.router)
    app.include_router(jobs.router)
    with TestClient(app) as client:
        yield client


def start(client: TestClient, body: dict[str, object]) -> str:
    response = client.post("/admin/benches", json=body)
    assert response.status_code == 202, response.text
    assert response.headers["location"].endswith(response.json()["id"])
    # The jobs screen filters by kind, so the string is a contract and not a label.
    assert response.json()["kind"] == "bench"
    job_id = response.json()["id"]
    assert isinstance(job_id, str)
    return job_id


def history(client: TestClient, model: str | None = None) -> dict[str, object]:
    response = client.get("/admin/benches", params=None if model is None else {"model": model})
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def measurement(value: object) -> dict[str, object]:
    assert isinstance(value, dict), f"expected a measurement, got {value!r}"
    return value


def rows(body: dict[str, object]) -> list[dict[str, object]]:
    benches = body["benches"]
    assert isinstance(benches, list), f"expected a list of measurements, got {benches!r}"
    return [measurement(value) for value in benches]


def number(row: dict[str, object], key: str) -> float:
    value = row[key]
    assert isinstance(value, int | float), f"expected a number at {key!r}, got {value!r}"
    return value


def bench_one(client: TestClient, model: str, rounds: int) -> None:
    finished = wait_for(client, start(client, {"model": model, "rounds": rounds}), "ok")
    assert finished["error"] is None


def test_a_bench_records_the_median_of_its_rounds_and_never_the_warmup(tmp_path: Path) -> None:
    """What is kept is the middle round and not the mean, which one bad round drags, and the
    warmup is not one of the rounds at all: it pays for the kernels it compiles and the
    first touch of the weights.

    The script separates the three readings that could be reported — median 200, mean 350,
    and 150 if the warmup were counted as a sixth round.
    """
    scripted = Scripted(
        [(1.0, 10.0), (100.0, 0.05), (500.0, 0.30), (200.0, 0.11), (900.0, 0.40), (50.0, 0.02)]
    )
    with stand(lambda _: composite(scripted), tmp_path / "server.db") as client:
        bench_one(client, "scripted", rounds=5)
        recorded = rows(history(client))

    assert scripted.runs == 6, "one warmup and five rounds"
    assert len(recorded) == 1
    assert recorded[0]["model"] == "scripted"
    assert number(recorded[0], "tokens_per_second") == pytest.approx(200.0)
    assert number(recorded[0], "ttft_ms") == pytest.approx(110.0)
    # No `nn.Module` under this one, so nobody counted its bytes per token: the row carries
    # no percentage rather than a percentage of a ceiling nobody computed.
    assert recorded[0]["ceiling_fraction"] is None


def test_the_percentage_of_the_ceiling_divides_by_the_models_active_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fraction is the whole reason a bench number means anything on this machine, and
    it has one right denominator: the bytes a decode step reads off this tree, over the
    490 GB/s a serial chain sustains. Divided by anything else — the theoretical 614, the
    whole checkpoint — the row reads plausible and is wrong."""
    monkeypatch.setattr(bench, "_TOKENS", 8)
    with stand(lambda _: tiny(), tmp_path / "server.db") as client:
        bench_one(client, "tiny", rounds=1)
        recorded = rows(history(client))

    assert len(recorded) == 1
    rate = number(recorded[0], "tokens_per_second")
    assert number(recorded[0], "ceiling_fraction") == pytest.approx(rate / TINY_CEILING)


def test_the_history_outlives_the_process_and_carries_the_engine_version(
    tmp_path: Path,
) -> None:
    """A bench is kept to be compared against a bench from before the machine was
    reinstalled, so the row has to survive the daemon that took it — and to be worth
    anything it has to say which engine took it. The second stand refuses to load
    anything: what it answers with came off the file."""
    scripted = Scripted([(1.0, 1.0), (300.0, 0.25)])
    path = tmp_path / "server.db"
    with stand(lambda _: composite(scripted), path) as client:
        bench_one(client, "scripted", rounds=1)
        before = rows(history(client))

    def refuses(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        raise AssertionError(f"the restarted daemon loaded {model_id!r} to read its history")

    with stand(refuses, path) as restarted:
        after = rows(history(restarted))

    assert len(after) == 1
    assert after == before
    assert after[0]["engine_version"] == version("mlx_omnia")
    assert number(after[0], "tokens_per_second") == pytest.approx(300.0)


def test_the_history_says_in_the_body_that_it_is_not_interleaved_against_mlx_lm(
    tmp_path: Path,
) -> None:
    """The one thing a reader of these numbers must not conclude is that mlx_omnia got faster
    or slower than mlx-lm. Interleaved A/B against mlx-lm git main is what counts a
    regression and it does not fit behind an endpoint, so the response says so itself: a
    docstring does not travel with the JSON someone pastes into a table.

    Read off a response that carries a measurement, because that is the one that gets
    pasted.
    """
    scripted = Scripted([(1.0, 1.0), (240.0, 0.20)])
    with stand(lambda _: composite(scripted), tmp_path / "server.db") as client:
        bench_one(client, "scripted", rounds=1)
        body = history(client)

    assert len(rows(body)) == 1
    method = body["method"]
    assert isinstance(method, str)
    assert "interleaved" in method
    assert "mlx-lm" in method
    assert "omnia-bench interleaved" in method, "the instrument is not named"


def test_a_delete_stops_the_bench_inside_a_round_and_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round is seconds of GPU. `asyncio.Task.cancel()` reaches neither the decode loop nor
    the thread driving it, and a flag read only between rounds would leave the machine
    generating tokens for a job the user already ended — so the wait on a round is broken up
    and the engine's own flag is set from inside it.

    A bench that was stopped is also a bench that was not measured: half a round has no
    median, and a row written from it would be a slow number nobody can explain later.
    """
    monkeypatch.setattr(bench, "_TOKENS", _LONG)
    paced = Paced()
    with stand(lambda _: composite(paced), tmp_path / "server.db") as client:
        job_id = start(client, {"model": "paced", "rounds": 3})
        assert paced.started.wait(_DEADLINE), "the bench never began generating"
        assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
        cancelled = wait_for(client, job_id, "cancelled")
        recorded = rows(history(client))

    assert cancelled["error"] is None
    assert paced.tokens < _LONG, "the round ran to the end and only then noticed"
    assert recorded == []


def test_a_model_that_does_not_load_fails_the_job_and_writes_no_row(tmp_path: Path) -> None:
    """The name comes off a text field, so a bench of a checkpoint that is not there is the
    ordinary mistake. It has to reach the client as the job's error: swallowed, it would be
    a bench that reports `ok` and a history that never grows."""

    def refuses(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        raise ValueError(f"no model {model_id!r} on this machine")

    with stand(refuses, tmp_path / "server.db") as client:
        died = wait_for(client, start(client, {"model": "ghost", "rounds": 1}), "error")
        recorded = rows(history(client))

    assert died["error"] == "ValueError: no model 'ghost' on this machine"
    assert recorded == []


def test_the_history_can_be_asked_for_one_model(tmp_path: Path) -> None:
    """The model card shows the last measurement of *its* model. Reading the whole history
    to draw one card is how a second model's number ends up under the first one's name."""
    scripted = {
        "fast": Scripted([(1.0, 1.0), (900.0, 0.10)]),
        "slow": Scripted([(1.0, 1.0), (90.0, 0.50)]),
    }
    with stand(lambda model_id: composite(scripted[model_id]), tmp_path / "server.db") as client:
        bench_one(client, "fast", rounds=1)
        bench_one(client, "slow", rounds=1)
        everything = rows(history(client))
        only_slow = rows(history(client, "slow"))

    assert [row["model"] for row in everything] == ["slow", "fast"], "newest first"
    assert [row["model"] for row in only_slow] == ["slow"]
    assert number(only_slow[0], "tokens_per_second") == pytest.approx(90.0)
