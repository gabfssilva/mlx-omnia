"""The speed runner: what it refuses before loading anything, what the ceiling divides by,
and what one shape produces.

The arithmetic is tested as arithmetic — it decides whether a shape runs at all, and it has
to be checkable without a checkpoint on the disk. The measurement is tested against the
same scripted model `test_bench.py` uses: the meter's marks are chosen instead of clocked,
so a median asserted here is a median over known numbers and not over whatever the machine
was doing while the suite ran.
"""

import asyncio
import json
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeIs

import pytest

from sideros import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    Model,
    ModelInput,
    ModelSignature,
    Text,
)
from sideros.parsers import Segment
from sideros_server import speed
from sideros_server.engine import Engine
from sideros_server.jobs import Bench, Job, JobView, Progress
from sideros_server.speed import ModelFacts, Sampling, SpeedShape, measure
from sideros_server.store import Store

GIGABYTE = 1024**3

DENSE = ModelFacts(
    weight_bytes=17 * GIGABYTE,
    kv_bytes_per_token=98_304,
    attention_window=None,
    checkpoint_bytes=17 * GIGABYTE,
)
"""A 30B mixture at 4 bits, in the house's own numbers: 1.7 GB read per step is the
mixture's slice, 96 KB per token is its cache."""

WINDOWED = replace(DENSE, attention_window=8192)

BF16 = replace(DENSE, weight_bytes=60 * GIGABYTE, checkpoint_bytes=60 * GIGABYTE)
"""The same architecture unquantized: the weights that have to be resident before the cache
gets any room are what turns a shape from tight into impossible."""

BF16_WINDOWED = replace(BF16, attention_window=8192)


def test_the_key_names_the_shape_and_never_the_model() -> None:
    shape = SpeedShape(context=4096, generate=256, concurrency=4)

    assert shape.key == "4k→256 · 4 streams · r3 · greedy · warm · queue"
    assert SpeedShape(context=1048576, generate=128, concurrency=1).key.startswith("1M→128")
    assert SpeedShape(context=512, generate=128, concurrency=1).key.startswith("512→128")


def test_the_same_shape_written_two_ways_is_one_key() -> None:
    """Which is the whole reason the three sets are closed."""
    first = SpeedShape(context=4096, generate=256, concurrency=1)
    second = SpeedShape(context=4096, generate=256, concurrency=1, rounds=3)

    assert first.key == second.key


def test_the_sampler_is_in_the_key_because_it_is_in_the_step() -> None:
    """A drawn token pays for filters an argmax does not, and two numbers taken under
    different samplers are not one axis."""
    argmax = SpeedShape(context=4096, generate=256, concurrency=1)
    drawn = replace(argmax, sampling=Sampling(temperature=0.7, top_p=0.95, top_k=40))

    assert argmax.key.endswith(" · greedy · warm · queue")
    assert " · t0.7·k40·p0.95 · " in drawn.key
    assert drawn.key != argmax.key
    # A knob left where it changes nothing does not split the axis.
    assert replace(argmax, sampling=Sampling(top_p=1.0, min_p=0.0)).key == argmax.key


def test_the_options_are_greedy_only_where_the_temperature_is_zero() -> None:
    assert Sampling().options(8).sampler is speed.greedy
    assert Sampling(temperature=0.7).options(8).sampler is not speed.greedy
    assert Sampling().options(8).penalty is None
    assert Sampling(repetition_penalty=1.1).options(8).penalty is not None


def test_more_than_one_stream_is_refused_until_the_scheduler_lands() -> None:
    """With the queue serialising generation the aggregate would be the scheduler's
    number. A row that says so beats a number that lies."""
    refused = speed.refusal(
        SpeedShape(context=4096, generate=256, concurrency=4), DENSE, 120 * GIGABYTE
    )

    assert refused is not None
    assert refused.reason == "concurrency_unsupported"
    assert refused.detail is not None and "45.3" in refused.detail


def test_a_cache_that_does_not_fit_is_a_refusal_with_both_numbers() -> None:
    shape = SpeedShape(context=1048576, generate=2048, concurrency=1)

    refused = speed.refusal(shape, BF16, 120 * GIGABYTE)

    assert refused is not None
    assert refused.reason == "kv_over_budget"
    assert refused.needed_bytes is not None and refused.needed_bytes > 120 * GIGABYTE
    assert refused.budget_bytes == 120 * GIGABYTE


def test_a_window_keeps_a_shape_that_full_attention_cannot_hold() -> None:
    """The same request, two checkpoints: the one that stops growing runs."""
    shape = SpeedShape(context=1048576, generate=2048, concurrency=1)

    assert speed.refusal(shape, BF16, 120 * GIGABYTE) is not None
    assert speed.refusal(shape, BF16_WINDOWED, 120 * GIGABYTE) is None


def test_the_ceiling_falls_as_the_context_rises() -> None:
    """The cache is the term that grows, so the same checkpoint has a lower ceiling at 32k
    than at 512 — and a fraction computed against the bytes of a zero-length context would
    not move at all."""
    short = SpeedShape(context=512, generate=256, concurrency=1)
    long = SpeedShape(context=32768, generate=256, concurrency=1)

    high = speed.ceiling_tps(short, DENSE)
    low = speed.ceiling_tps(long, DENSE)

    assert high is not None and low is not None
    assert low < high
    assert speed.kv_step_bytes(long, DENSE) > speed.kv_step_bytes(short, DENSE)


def test_the_window_stops_the_cache_from_growing_past_it() -> None:
    beyond = SpeedShape(context=32768, generate=256, concurrency=1)

    assert speed.kv_step_bytes(beyond, WINDOWED) == 98_304 * 8192
    assert speed.ceiling_tps(beyond, WINDOWED) == speed.ceiling_tps(
        SpeedShape(context=65536, generate=256, concurrency=1), WINDOWED
    )


def test_a_checkpoint_with_no_priced_weights_has_no_ceiling() -> None:
    """Not a zero, and not a guess."""
    unpriced = replace(DENSE, weight_bytes=None)

    assert speed.ceiling_tps(SpeedShape(context=512, generate=128, concurrency=1), unpriced) is None


def test_the_percentile_is_a_round_that_was_measured() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    assert speed.percentile(values, 0.5) in values
    assert speed.percentile(values, 0.95) == 8.0
    assert speed.percentile([4.0], 0.95) == 4.0


@dataclass
class Scripted:
    """A generation whose numbers are chosen instead of clocked, as in `test_bench.py`. The
    script walking off its end is how a round nobody asked for fails the test it is in."""

    script: list[tuple[float, float]]
    """(decode tok/s, ttft in seconds) per generation, the load probe first."""

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
        meter.prompt_tokens = len(input.value) // 4
        meter.completion_tokens = options.max_tokens
        meter.prefill_started = 0.0
        meter.first_token = ttft
        meter.last_token = ttft + (options.max_tokens - 1) / rate
        for index in range(options.max_tokens):
            yield Segment("content", str(index))


def composite(
    model: Model[Text, Segment, GenerationOptions],
) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(model, [])


def _job(loop: asyncio.AbstractEventLoop, store: Store) -> Job:
    view = JobView(
        id="bench",
        kind="bench",
        subject=Bench(model="m"),
        state="running",
        progress=Progress(),
        created_at=0.0,
        updated_at=0.0,
    )
    return Job(view=view, loop=loop, store=store)


@pytest.fixture
def stand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[Engine, Job]]:
    """An engine on a loop of its own thread, which is where `measure` expects to find one:
    it runs in a worker thread and drives the loop through `run_coroutine_threadsafe`."""
    monkeypatch.setattr(speed, "macmon", lambda: None)
    model = Scripted(script=[(50.0, 0.5), *[(100.0 + index, 0.2) for index in range(10)]])
    engine = Engine(lambda _model_id: composite(model))
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def spin() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(engine.start)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    ready.wait(5)
    store = Store(tmp_path / "server.db")
    try:
        yield engine, _job(loop, store)
    finally:
        loop.call_soon_threadsafe(engine.stop)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def test_a_shape_produces_four_measurements_and_drops_the_warm_up(
    stand: tuple[Engine, Job],
) -> None:
    engine, job = stand
    shape = SpeedShape(context=512, generate=64, concurrency=1, rounds=4)

    taken = measure(job, engine, "m", shape, DENSE, 120 * GIGABYTE, warm_up=True)

    assert taken.state == "ok"
    assert taken.result.load_s is not None and taken.result.load_s >= 0
    assert taken.result.prefill_tps is not None and taken.result.prefill_tps > 0
    assert taken.result.ttft_p50_ms == pytest.approx(200.0)
    assert taken.result.decode_tps is not None
    # The script's first two generations are the load probe and the warm-up; the four kept
    # rounds are 101, 102, 103 and 104 tok/s, whose median is 102.5.
    assert taken.result.decode_tps == pytest.approx(102.5)
    assert len(json.loads(taken.result.per_round)) == 4


def test_the_recorded_ceiling_is_the_one_the_fraction_divides_by(
    stand: tuple[Engine, Job],
) -> None:
    """Both bytes are on the row, so the division can be redone by hand."""
    engine, job = stand
    shape = SpeedShape(context=512, generate=64, concurrency=1, rounds=2)

    taken = measure(job, engine, "m", shape, DENSE, 120 * GIGABYTE)

    assert taken.result.step_weight_bytes is not None
    assert taken.result.step_kv_bytes is not None
    step = taken.result.step_weight_bytes + taken.result.step_kv_bytes
    assert taken.result.ceiling_tps == pytest.approx(speed.BANDWIDTH_GBS * 1e9 / step)
    assert taken.result.decode_tps is not None and taken.result.ceiling_tps is not None
    assert taken.result.ceiling_fraction == pytest.approx(
        taken.result.decode_tps / taken.result.ceiling_tps
    )


def test_a_refused_shape_never_reaches_the_engine(stand: tuple[Engine, Job]) -> None:
    """No model is loaded to find out that its cache does not fit."""
    engine, job = stand
    shape = SpeedShape(context=1048576, generate=2048, concurrency=1)

    taken = measure(job, engine, "m", shape, BF16, 120 * GIGABYTE)

    assert taken.state == "not_run"
    assert taken.reason == "kv_over_budget"
    assert engine.resident == []
    detail = json.loads(taken.result.per_round)
    assert detail["needed_bytes"] > detail["budget_bytes"]
    assert taken.result.context == 1048576


def test_a_cold_page_cache_that_cannot_be_produced_is_not_reported_as_cold(
    stand: tuple[Engine, Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`purge(8)` needs root on a current system. A cold measurement taken on a warm cache
    is a warm measurement wearing the wrong label."""
    engine, job = stand
    monkeypatch.setattr(speed, "purge_page_cache", lambda: False)
    shape = SpeedShape(context=512, generate=64, concurrency=1, rounds=1, page_cache="cold")

    taken = measure(job, engine, "m", shape, DENSE, 120 * GIGABYTE)

    assert taken.state == "not_run"
    assert taken.reason == "page_cache_unavailable"


def test_the_gate_records_the_temperature_the_round_started_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """And waits while the machine is above it."""
    readings = iter([70.0, 60.0, 38.0])
    monkeypatch.setattr(speed, "gpu_temperature", lambda _tool: next(readings))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    loop = asyncio.new_event_loop()
    job = _job(loop, Store(tmp_path / "server.db"))

    assert speed.wait_cool(job, "macmon", 40.0) == 38.0
    loop.close()


def test_without_a_temperature_source_the_gate_does_not_invent_one(tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    job = _job(loop, Store(tmp_path / "server.db"))

    assert speed.wait_cool(job, None, 40.0) is None
    loop.close()
