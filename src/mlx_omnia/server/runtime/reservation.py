"""The queue held exclusively, for as long as a measurement runs."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from mlx_omnia.server.runtime.rounds import Rounds


class Reserving(Rounds):
    async def acquire_queue(self) -> object:
        """The queue, exclusively, until the token comes back.

        A measurement is only worth what nothing else running beside it is worth, and the gate
        alone does not give that: it serialises generation, so a chat request lands *between* two
        rounds and moves the median without appearing anywhere. Holding the queue is what closes
        it. Everybody else waits in `submit`, before being queued at all, so nothing is dropped
        and the ordering stays the queue's own.
        """
        await self._reserving.acquire()
        token = object()
        self._reserved = token
        self._free.clear()
        self.on_change()
        # What was already queued when the reservation was taken is still somebody's request, and
        # it is measured against nothing: let the worker finish it before the first round starts.
        while self._current or self._pending or not self._queue.empty():
            await asyncio.sleep(0.01)
        return token

    async def release_queue(self, token: object) -> None:
        """Idempotent, and a no-op for a token that is not the holder's: the release rides a
        `finally`, and a benchmark that raised after the daemon was already restarted must not
        hand somebody else's reservation back."""
        if self._reserved is not token:
            return
        self._reserved = None
        self._free.set()
        self._reserving.release()
        self.on_change()

    @asynccontextmanager
    async def reserve(self) -> AsyncGenerator[object]:
        """`acquire_queue` and `release_queue` as a block, for a caller that lives on the loop.
        The work of a job does not — it runs in a thread and drives the loop through
        `run_coroutine_threadsafe` — which is why the two halves exist separately."""
        token = await self.acquire_queue()
        try:
            yield token
        finally:
            await self.release_queue(token)
