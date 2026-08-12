"""What the views need from the runtime that Flet does not already hand them."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine

import flet as ft


def use_tick(period: float = 1.0) -> int:
    """A counter that advances every `period` seconds.

    Text that ages — "for 12 s", "3 min ago" — has no event to redraw on, because nothing
    changed but the clock. A component that shows one reads this, and only that component
    wakes up on the beat.
    """
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


_SHEETS: list[Callable[[], Callable[[], None] | None]] = []
"""The screens that can put a sheet over themselves, each able to say what closes the
innermost one it has open."""


def sheets(ask: Callable[[], Callable[[], None] | None]) -> Callable[[], None]:
    """Register one, and get back the unregister — which is what `use_effect` wants.

    Asked rather than pushed and popped: a sheet closes from more than one place — a run
    starting takes the Benchmark's down without anybody pressing anything — and a stack
    maintained at every one of those is a stack that goes wrong at the one somebody forgets.
    """
    _SHEETS.append(ask)
    return lambda: _SHEETS.remove(ask)


def escape() -> bool:
    """Close the innermost open sheet, and say whether there was one.

    Only the innermost answers: Library can open Quantize over itself, and the Benchmark's
    sheet opens the dataset form over its own.
    """
    for ask in reversed(_SHEETS):
        close = ask()
        if close is not None:
            close()
            return True
    return False


def act(work: Coroutine[None, None, None]) -> None:
    """Run an action the screen started, and keep a refusal from taking the app with it.

    The daemon's "no" is an answer, not a crash: every call the views make can be refused,
    and what the screen does about it is show the message the next frame carries.
    """

    async def guarded() -> None:
        try:
            await work
        except Exception as error:  # noqa: BLE001 — a refused action is not a crash
            print(f"action failed: {error}")

    ft.context.page.run_task(guarded)
