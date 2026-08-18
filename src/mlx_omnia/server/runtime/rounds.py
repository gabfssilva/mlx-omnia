"""The FCFS worker: one item off the queue, the group it opens, and the clock it runs on."""

import asyncio
import time
from collections.abc import Sequence
from typing import TypeIs

import mlx.core as mx

from mlx_omnia.engine.language import ContinuousLanguageModel, TextBatch
from mlx_omnia.engine.model import CompositeModel
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime.decoding import Phases
from mlx_omnia.server.runtime.flow import Clock, Emission
from mlx_omnia.server.runtime.jobs import (
    Decoding,
    Generation,
    GenerationEmission,
    GenerationMember,
    Job,
    Prefilling,
    Release,
    Streaming,
)


class Rounds(Phases):
    def _concurrency(self, job: Job) -> int:
        if job.batch_limit is not None:
            return job.batch_limit
        if self._environment is None:
            return 1
        return self._environment.concurrency(job.model_id)

    async def _run(self) -> None:
        while True:
            item = self._pending.popleft() if self._pending else await self._queue.get()
            if isinstance(item, Release):
                item.model = None
                # Back to the system rather than into MLX's own buffer cache: what an unload is
                # for is a footprint that came down, and the footprint is what the memory rail
                # reads and what admission decides against.
                mx.clear_cache()
                item.done.set()
                continue
            await self._round(item)
            # A round's jobs die with its frame; this name is the last reference left, and holding
            # it across the wait above would keep a model alive through its unload.
            del item

    async def _round(self, item: Job) -> None:
        """One group, from the queue to the last token: whoever else is queued for the same model
        and batcher joins it, and every member is metered whether it generates or was already
        cancelled."""
        jobs = [item]
        batcher = self._batcher(item)
        concurrency = self._concurrency(item)
        if batcher is not None:
            await asyncio.sleep(0)
            while len(jobs) < concurrency:
                try:
                    candidate = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if self._joins(candidate, item, batcher):
                    jobs.append(candidate)
                    continue
                self._pending.append(candidate)
                break
        self._current = jobs.copy()
        self.on_change()
        for job in jobs:
            entry = self._residency.get(job.model_id)
            job.metrics_key = self._metrics.begin(
                job.model_id,
                job.meter,
                None if entry is None else entry.active_bytes,
                job.load_seconds,
            )
        try:
            active = [job for job in jobs if not job.cancelled.is_set()]
            for job in jobs:
                if job not in active:
                    job.state = "cancelled"
                    job.chunks.put_nowait(None)
            if active:
                await self._generate(batcher, active, jobs)
        except Exception as error:
            for job in jobs:
                if job.state not in ("completed", "cancelled"):
                    job.error = f"{type(error).__name__}: {error}"
                    job.state = "error"
                    job.chunks.put_nowait(None)
        finally:
            self._current.clear()
            for job in jobs:
                self._metrics.end(job.state, job.metrics_key)
                if job.lease is not None:
                    job.lease.leases -= 1
            self._changed.set()
            self.on_change()

    def _joins(
        self, candidate: object, exemplar: Job, batcher: ContinuousLanguageModel
    ) -> TypeIs[Job]:
        """Whether a queued item can ride the group `exemplar` opened: the same resident model,
        and a batcher that is the very object already decoding — two ids pointing at one
        checkpoint are one model, and two models are never one batch."""
        return (
            isinstance(candidate, Job)
            and candidate.model is exemplar.model
            and self._batcher(candidate) is batcher
        )

    @staticmethod
    def _batcher(job: Job) -> ContinuousLanguageModel | None:
        model = job.model
        if not isinstance(model, CompositeModel):
            return None
        return (
            model.model
            if isinstance(model.model, ContinuousLanguageModel)
            and model.model.can_batch(job.options)
            else None
        )

    def _next_batched(self, model: ContinuousLanguageModel, exemplar: Job) -> Job | None:
        if self._pending:
            candidate = self._pending[0]
            if self._joins(candidate, exemplar, model):
                self._pending.popleft()
                return candidate
            return None
        try:
            candidate = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if self._joins(candidate, exemplar, model):
            return candidate
        self._pending.append(candidate)
        return None

    async def _generate(
        self, model: ContinuousLanguageModel | None, initial: Sequence[Job], all_jobs: list[Job]
    ) -> None:
        """One group, from the first prompt to the last token, on one clock.

        A batchable group's tick is three things in order: one prefill block for every member
        still feeding its prompt, one shared `step_batch` over the members that are decoding, and
        whatever joiners the concurrency has room for — admitted as prompts being fed, never as a
        whole prefill run in front of everybody else's decode.

        A group the batcher refused is the same clock with one member in it and no room for a
        second: its tick is one `next` on the generator `stream` returned.
        """
        entry = self._residency.get(initial[0].model_id)
        peak = 0

        def tick(members: Sequence[GenerationMember]) -> list[GenerationEmission]:
            nonlocal peak
            emissions: list[GenerationEmission] = []
            decoding: list[tuple[GenerationMember, TextBatch]] = []
            held = 0
            for member in members:
                phase = member.state.phase
                match phase:
                    case Prefilling(prefill=prefill):
                        batch = prefill.advance()
                        if batch is not None:
                            member.state.phase = Decoding(batch)
                        emissions.append(Emission(member))
                    case Decoding(batch=batch):
                        held += sum(layer.nbytes for layer in batch.state.cache)
                        decoding.append((member, batch))
                    case Streaming(pieces=pieces):
                        piece = next(pieces, None)
                        emissions.append(
                            Emission(
                                member,
                                # Routing, nobody's prose: a header names the channel the next
                                # text rides, and no API dialect has a field for it. Dropped here
                                # once, so the four dialects never see one.
                                () if piece is None or piece.channel == "header" else (piece,),
                                piece is None,
                            )
                        )
            peak = max(peak, held)
            if entry is not None and decoding:
                entry.kv_bytes = held
            if not decoding:
                return emissions
            assert model is not None, "a group with no batcher never reaches a decoding phase"
            produced = model.step_batch([batch for _, batch in decoding])
            for (member, batch), pieces in zip(decoding, produced, strict=True):
                emissions.append(
                    Emission(
                        member,
                        tuple(piece for piece in pieces if piece.channel != "header"),
                        batch.state.finished,
                    )
                )
            return emissions

        def room() -> int:
            return 0 if model is None else self._concurrency(initial[0])

        def waiting() -> bool:
            """Whether anything is queued at all — a deque and a queue, no store behind either.
            `room` is what reads the limits, and this is what keeps it off the path a token
            takes."""
            return bool(self._pending) or not self._queue.empty()

        async def join() -> GenerationMember | None:
            assert model is not None, "no room is admitted for a group with no batcher"
            joining = self._next_batched(model, initial[0])
            if joining is None:
                return None
            all_jobs.append(joining)
            self._current.append(joining)
            record = self._residency.get(joining.model_id)
            joining.metrics_key = self._metrics.begin(
                joining.model_id,
                joining.meter,
                None if record is None else record.active_bytes,
                joining.load_seconds,
            )
            if joining.cancelled.is_set():
                joining.state = "cancelled"
                joining.chunks.put_nowait(None)
                return None
            member = await self._enlist(model, joining)
            self.on_change()
            return member

        def leave(member: GenerationMember) -> None:
            job = member.state.job
            phase = member.state.phase
            if isinstance(phase, Streaming):
                self._account(job.model_id, phase.settled, phase.loads)
            # Before the sentinel: what the consumer reads after it is a terminal state.
            job.state = "cancelled" if job.cancelled.is_set() else "completed"

        clock: Clock[Generation, Segment] = Clock(
            self._model_thread, tick, room=room, waiting=waiting, join=join, on_leave=leave
        )
        members = [await self._enlist(model, job) for job in initial]
        try:
            await clock.run(members)
        finally:
            if model is not None and entry is not None:
                entry.kv_bytes = peak
                entry.last_used = time.time()
