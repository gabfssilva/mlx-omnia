"""What one shape produces, and what the gate records before it.

The measurement is tested against a scripted model: the meter's marks are chosen instead of
clocked, so a median asserted here is a median over known numbers and not over whatever the
machine was doing while the suite ran. The arithmetic that decides whether a shape runs at
all is `test_speed_shapes.py`.

`measure` and `wait_cool` bind the names they use at import, so the thermal doubles are
patched into the modules that call them and not only into the package that re-exports them.
"""

import asyncio
import json
import threading
import time
from collections.abc import Generator
from importlib import import_module

import pytest

from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services import speed
from mlx_omnia.server.services.speed import SpeedShape, measure

from .speed_stand import (
    BF16,
    DENSE,
    GIGABYTE,
    Handle,
    Scripted,
    composite,
)

_MEASURE = import_module("mlx_omnia.server.services.speed.measure")
_THERMAL = import_module("mlx_omnia.server.services.speed.thermal")


@pytest.fixture
def stand(monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[Engine, Handle]]:
    """An engine on a loop of its own thread, which is where `measure` expects to find one:
    it runs in a worker thread and drives the loop through `run_coroutine_threadsafe`."""
    monkeypatch.setattr(_MEASURE, "macmon", lambda: None)
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
    try:
        yield engine, Handle(loop=loop)
    finally:
        loop.call_soon_threadsafe(engine.stop)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)


def test_a_shape_produces_four_measurements_and_drops_the_warm_up(
    stand: tuple[Engine, Handle],
) -> None:
    engine, task = stand
    shape = SpeedShape(context=512, generate=64, concurrency=1, rounds=4)

    taken = measure(task, engine, "m", shape, DENSE, 120 * GIGABYTE, warm_up=True)

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
    stand: tuple[Engine, Handle],
) -> None:
    """Both bytes are on the row, so the division can be redone by hand."""
    engine, task = stand
    shape = SpeedShape(context=512, generate=64, concurrency=1, rounds=2)

    taken = measure(task, engine, "m", shape, DENSE, 120 * GIGABYTE)

    assert taken.result.step_weight_bytes is not None
    assert taken.result.step_kv_bytes is not None
    step = taken.result.step_weight_bytes + taken.result.step_kv_bytes
    assert taken.result.ceiling_tps == pytest.approx(speed.BANDWIDTH_GBS * 1e9 / step)
    assert taken.result.decode_tps is not None and taken.result.ceiling_tps is not None
    assert taken.result.ceiling_fraction == pytest.approx(
        taken.result.decode_tps / taken.result.ceiling_tps
    )


def test_a_refused_shape_never_reaches_the_engine(stand: tuple[Engine, Handle]) -> None:
    """No model is loaded to find out that its cache does not fit."""
    engine, task = stand
    shape = SpeedShape(context=1048576, generate=2048, concurrency=1)

    taken = measure(task, engine, "m", shape, BF16, 120 * GIGABYTE)

    assert taken.state == "not_run"
    assert taken.reason == "kv_over_budget"
    assert engine.resident == []
    detail = json.loads(taken.result.per_round)
    assert detail["needed_bytes"] > detail["budget_bytes"]
    assert taken.result.context == 1048576


def test_a_cold_page_cache_that_cannot_be_produced_is_not_reported_as_cold(
    stand: tuple[Engine, Handle], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`purge(8)` needs root on a current system. A cold measurement taken on a warm cache
    is a warm measurement wearing the wrong label."""
    engine, task = stand
    monkeypatch.setattr(_MEASURE, "purge_page_cache", lambda: False)
    shape = SpeedShape(context=512, generate=64, concurrency=1, rounds=1, page_cache="cold")

    taken = measure(task, engine, "m", shape, DENSE, 120 * GIGABYTE)

    assert taken.state == "not_run"
    assert taken.reason == "page_cache_unavailable"


def test_the_gate_records_the_temperature_the_round_started_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And waits while the machine is above it."""
    readings = iter([70.0, 60.0, 38.0])
    monkeypatch.setattr(_THERMAL, "gpu_temperature", lambda _tool: next(readings))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    loop = asyncio.new_event_loop()

    assert speed.wait_cool(Handle(loop=loop), "macmon", 40.0) == 38.0
    loop.close()


def test_without_a_temperature_source_the_gate_does_not_invent_one() -> None:
    loop = asyncio.new_event_loop()

    assert speed.wait_cool(Handle(loop=loop), None, 40.0) is None
    loop.close()
