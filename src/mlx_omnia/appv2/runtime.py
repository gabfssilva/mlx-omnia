"""What the appv2 screens need from the runtime beyond Flet: guarded actions, the
decode trace, and the newest numbers a model has to show.

The trace lives outside any view for the v1 reason — the daemon keeps no history to
ask for, so the window records one for as long as it runs, whichever screen is on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine

import flet as ft

from mlx_omnia.app.api.engine import Engine, Sample


def use_tick(period: float = 1.0) -> int:
    """A counter advancing every `period` seconds — the redraw for text that only ages
    and for the pulse, which moves while nothing else does."""
    beat, set_beat = ft.use_state(0)

    def spin() -> Callable[[], None]:
        async def advance() -> None:
            count = beat
            while True:
                await asyncio.sleep(period)
                count += 1
                set_beat(count)

        task = asyncio.get_running_loop().create_task(advance())
        return task.cancel

    ft.use_effect(spin, [period])
    return beat


def act(work: Coroutine[None, None, None]) -> None:
    """Run an action a screen started, and keep a refusal from taking the app with it."""

    async def guarded() -> None:
        try:
            await work
        except Exception as error:  # noqa: BLE001 — a refused action is not a crash
            print(f"action failed: {error}")

    ft.context.page.run_task(guarded)


TRACE = 56
_TRACE: dict[str, list[float]] = {}


async def trace(engine: Engine) -> None:
    """Sample every model's share of the ceiling, once a second, for as long as the app
    runs — the Server card's pulse."""
    while True:
        snapshot = engine.metrics
        if snapshot is not None:
            for aggregate in snapshot["models"]:
                model = aggregate["model"]
                live = engine.live(model)
                point = (live["ceiling_fraction"] or 0.0) if live is not None else 0.0
                _TRACE[model] = [*_TRACE.get(model, []), point * 100][-TRACE:]
        await asyncio.sleep(1.0)


def points(model: str) -> list[float]:
    return _TRACE.get(model, [])


def resident_ids(engine: Engine) -> set[str]:
    """Residency off the state stream, which a load republishes — the catalog's
    `resident` flag is only refreshed when the disk moves, and goes stale on a load."""
    return {slot["id"] for slot in engine.models}


def sample_of(engine: Engine, model: str) -> Sample | None:
    """The newest numbers this model has: the live register while it generates, else the
    most recent completed request. The wire carries requests newest first."""
    if engine.metrics is None:
        return None
    live = engine.live(model)
    if live is not None:
        return live
    for request in engine.metrics["requests"]:
        if request["model"] == model and request["state"] == "completed":
            return request
    return None


def decode_of(engine: Engine, model: str) -> str | None:
    sample = sample_of(engine, model)
    speed = None if sample is None else sample["tokens_per_second"]
    if speed is None:
        aggregate = engine.aggregate(model)
        speed = None if aggregate is None else aggregate["tokens_per_second"]
    return None if speed is None else f"{speed:.1f} tok/s"


def prefill_ttft_of(engine: Engine, model: str) -> str | None:
    sample = sample_of(engine, model)
    if sample is None:
        return None
    prefill, ttft = sample["prefill_tokens_per_second"], sample["ttft"]
    if prefill is None and ttft is None:
        return None
    left = "—" if prefill is None else f"{prefill:,.0f}"
    right = "—" if ttft is None else f"{ttft * 1000:.0f} ms"
    return f"{left} prefill · {right}"
