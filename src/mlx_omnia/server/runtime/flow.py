"""The one loop a generation lives inside, and nothing about generation in it.

A `Clock` owns membership and routing: who is in the group, who leaves, and where the pieces a
tick produced go. What a tick *is* belongs to whoever built the clock.
"""

import asyncio
import threading
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass


class Outlet[T]:
    """Where one member's pieces come out.

    `emit` never blocks the clock — the queue is unbounded — and `close` is idempotent and
    writes the `None` sentinel the queue's consumer ends on. Both are callable from the model
    thread: the queue belongs to the loop, so every write goes through `call_soon_threadsafe`.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[T | None]) -> None:
        self._loop = loop
        self._queue = queue
        self._closed = False

    def emit(self, item: T) -> None:
        if self._closed:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)


@dataclass
class Member[S, T]:
    """One request inside the clock: whatever state its owner keeps for it, the outlet its
    pieces leave through, and the flag that takes it out of the group."""

    state: S
    outlet: Outlet[T]
    cancelled: threading.Event


@dataclass(frozen=True)
class Emission[S, T]:
    """What one member got out of one tick."""

    member: Member[S, T]
    pieces: tuple[T, ...] = ()
    finished: bool = False


class Clock[S, T]:
    """The single loop over a group of members.

    Each turn: whoever was cancelled leaves, the tick runs on the model thread — through the
    caller's own executor, never a generic `to_thread`, because the Metal context lives on one
    specific thread — its emissions are routed, whoever finished leaves, and joiners are
    admitted while there is room.
    """

    def __init__(
        self,
        executor: Executor,
        tick: Callable[[Sequence[Member[S, T]]], Sequence[Emission[S, T]]],
        *,
        room: Callable[[], int],
        waiting: Callable[[], bool],
        join: Callable[[], Awaitable[Member[S, T] | None]],
        on_leave: Callable[[Member[S, T]], None],
    ) -> None:
        self._executor = executor
        self._tick = tick
        self._room = room
        self._waiting = waiting
        self._join = join
        self._on_leave = on_leave

    async def run(self, members: Sequence[Member[S, T]]) -> None:
        loop = asyncio.get_running_loop()
        active = list(members)
        while active:
            active = [member for member in active if not self._left(member)]
            if not active:
                break
            emissions = await loop.run_in_executor(self._executor, self._tick, tuple(active))
            done: set[int] = set()
            for emission in emissions:
                for piece in emission.pieces:
                    emission.member.outlet.emit(piece)
                if emission.finished:
                    self._leave(emission.member)
                    done.add(id(emission.member))
            active = [member for member in active if id(member) not in done]
            # A joiner queued while this tick ran is only visible to the queue's own consumer
            # after the loop has had a turn.
            await asyncio.sleep(0)
            # `waiting` before `room`: how much room there is comes out of the store, and
            # asking it once per token — for a queue empty on every one of them — puts two
            # SQLite reads on the decode's critical path.
            while self._waiting() and len(active) < self._room():
                joiner = await self._join()
                if joiner is None:
                    break
                active.append(joiner)

    def _left(self, member: Member[S, T]) -> bool:
        if not member.cancelled.is_set():
            return False
        self._leave(member)
        return True

    def _leave(self, member: Member[S, T]) -> None:
        self._on_leave(member)
        member.outlet.close()
