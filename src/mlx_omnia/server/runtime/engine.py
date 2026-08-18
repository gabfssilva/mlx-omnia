"""Resident models plus the global FCFS generation scheduler.

One worker owns a dedicated MLX thread. Compatible requests for the same model share decode
steps up to the configured concurrency; incompatible requests and transitions between models
remain FCFS. Reading a checkpoint gets a thread of its own — see `_load_thread`: one thread for
both is what lets a load wait on a generation that is itself waiting on the load, and what makes
a cold model's read stall every resident model's decode.

Nothing is resident at boot. A request names its model and that is what loads it, and nothing
but the memory limit takes it away again: a load that would cross the ceiling evicts the least
recently used model first, and one that has been idle past its TTL leaves on its own. Both
figures come from the environment, read per decision.
"""

import asyncio
import time
from dataclasses import replace

from mlx_omnia import GenerationOptions, ModelInput, UnsupportedInput
from mlx_omnia.server.runtime.errors import (
    ModelTooLarge,
    NotConstrainable,
    NotQuantizable,
    NotResident,
)
from mlx_omnia.server.runtime.jobs import Job, Release
from mlx_omnia.server.runtime.reservation import Reserving
from mlx_omnia.server.runtime.residency import Residency
from mlx_omnia.server.runtime.state import SPILL_JOIN_SECONDS, Loader
from mlx_omnia.server.runtime.walks import drafter, tree

__all__ = [
    "Engine",
    "Job",
    "Loader",
    "ModelTooLarge",
    "NotConstrainable",
    "NotQuantizable",
    "NotResident",
    "Residency",
    "drafter",
    "tree",
]


class Engine(Reserving):
    """Residency and the generation scheduler."""

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._worker = loop.create_task(self._run())
        self._sweeper = loop.create_task(self._sweep())

    async def stop(self) -> None:
        """Awaited, and not because anything here is slow: the two steps at the end run on
        threads, and they need this loop free while they do.

        A vault's index is async and its writer thread reaches it by handing coroutines back
        here (`services.prefixes.FileVault`). A `stop` that blocked the loop to wait for that
        thread would be waiting for a thread waiting for it — measured as a 30-second join
        that timed out, a `CancelledError` in `prefix-vault` when the database closed under it,
        and a payload left on disk with no row pointing at it.
        """
        if self._sweeper is not None:
            # Before the worker, so nothing new reaches a queue that is about to be drained.
            self._sweeper.cancel()
            self._sweeper = None
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        for job in self._current:
            job.cancel()
            job.chunks.put_nowait(None)
        self._current.clear()
        while self._pending:
            queued = self._pending.popleft()
            if isinstance(queued, Release):
                queued.model = None
                queued.done.set()
                continue
            queued.state = "cancelled"
            queued.chunks.put_nowait(None)
        while not self._queue.empty():
            queued = self._queue.get_nowait()
            if isinstance(queued, Release):
                # The models are cleared below anyway; what this line is for is the `unload`
                # waiting on the other side, which would otherwise wait for ever.
                queued.model = None
                queued.done.set()
                continue
            queued.state = "cancelled"
            queued.chunks.put_nowait(None)
        for task in self._loading.values():
            task.cancel()
        self._loading.clear()
        # Before the models go: the store outlives them, but the anchors it holds are only
        # written on the way out, and a shutdown that skipped this would lose exactly the
        # conversations somebody is in the middle of.
        if self._prefixes is not None:
            await asyncio.to_thread(self._prefixes[1].drain)
        self._models.clear()
        self._residency.clear()
        # A write in flight is a file half written under a staging name and a row that will never
        # be inserted. Nothing else waits for it — that is the whole point of writing behind the
        # request — so shutdown is where it is waited for.
        for _, vault in self._vaults.values():
            if vault is not None:
                await asyncio.to_thread(vault.flush, SPILL_JOIN_SECONDS)
        self._vaults.clear()
        self._model_thread.shutdown(wait=False, cancel_futures=True)
        self._load_thread.shutdown(wait=False, cancel_futures=True)

    async def submit(
        self,
        model_id: str,
        input: ModelInput,
        options: GenerationOptions,
        reservation: object | None = None,
        batch_limit: int | None = None,
    ) -> Job:
        """Raises `UnsupportedInput` before queueing: a model that cannot take this input never
        becomes a job, so the caller answers with a client error instead of the worker failing
        mid-generation. `NotResident` comes before even that, when the config says so.
        """
        # Before anything else, including the load: a request that waits here has not taken a
        # lease, has not moved a model's `last_used`, and has not put a byte on the GPU while
        # somebody is measuring.
        while self._reserved is not None and reservation is not self._reserved:
            await self._free.wait()
        # Cold decided before the await, because after it the model is resident either way. What
        # is timed is the whole wait — admission and the eviction it may order included: they are
        # the request's seconds too.
        cold = model_id not in self._models
        started = time.perf_counter()
        model = await self._reachable(model_id)
        loaded = time.perf_counter() - started if cold else None
        if not model.accepts(input):
            raise UnsupportedInput(input)
        entry = self._residency[model_id]
        entry.last_used = time.time()
        # The lease is taken here and nowhere else. `resolve` returns without suspending for a
        # model that is already resident, and the load it awaits ends with the entry written, so
        # no admission runs between the entry being found and being held.
        entry.leases += 1
        try:
            model = await self._compress(model_id, model, entry)
        except NotQuantizable:
            # The lease is taken before the policy is applied because `_compress` awaits, and an
            # await between the record being found and being held is one an eviction can use to
            # drop it. A refusal queues no job, so nothing else will ever give this lease back.
            entry.leases -= 1
            self._changed.set()
            raise
        # How much of what it reads the resident models may keep for the next request. It rides in
        # the options because out here there is no prompt yet — only a conversation — and it is a
        # ceiling rather than the cache because the cache's element type is the trunk's own, which
        # nothing holding a `LanguageModel[ModelInput]` can name.
        options = replace(options, prefix=self._prefixing(model_id))
        job = Job(
            model_id,
            model,
            input,
            options,
            asyncio.get_running_loop(),
            lease=entry,
            load_seconds=loaded,
            batch_limit=batch_limit,
        )
        await self._queue.put(job)
        self.on_change()
        return job
