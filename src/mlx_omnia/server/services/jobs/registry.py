from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from mlx_omnia.server.db.models.jobs import JobState
from mlx_omnia.server.services.jobs.views import (
    JobView,
    Progress,
    Subject,
    abandon,
    kind_of,
    view,
    write,
)

_WORKERS = 4
"""How many job bodies run at once. What they do is disk and network, where a handful of
concurrent streams already has the device; the gate that serializes generation is the
engine's own and is depth one."""


class Cancelled(Exception):
    """Raised inside the worker thread, by `Job.report`, when the job was cancelled."""


class NoSuchJob(Exception):
    """No job — live or persisted — answers to that id."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"no job {job_id!r}")
        self.job_id = job_id


class JobFinished(Exception):
    """A cancellation reached a job nobody is running any more."""

    def __init__(self, job_id: str, state: JobState) -> None:
        super().__init__(f"job {job_id!r} already finished as {state}")
        self.job_id = job_id
        self.state = state


def _nothing() -> None:
    pass


@dataclass
class Job:
    """The half of a job that cannot be a row: the flag the blocking work reads and the
    streams waiting on it."""

    view: JobView
    loop: asyncio.AbstractEventLoop
    writes: asyncio.Queue[JobView]
    cancelled: threading.Event = field(default_factory=threading.Event)
    watchers: set[asyncio.Queue[JobView | None]] = field(default_factory=set)
    announce: Callable[[], None] = _nothing

    @property
    def id(self) -> str:
        return self.view.id

    def cancel(self) -> None:
        self.cancelled.set()

    def report(self, progress: Progress) -> None:
        """The work's only door, and so the one place cancellation is found: a flag read
        between steps never interrupts the step itself."""
        if self.cancelled.is_set():
            raise Cancelled(self.view.id)
        self.publish(replace(self.view, state="running", progress=progress))

    def publish(self, current: JobView) -> None:
        """Called from whichever thread the work runs on: the row and the fanout both go
        to the loop, in that order, through one callback."""
        self.view = replace(current, updated_at=time.time())
        self.loop.call_soon_threadsafe(self._deliver, self.view)
        self.announce()

    def finish(self, state: JobState, error: str | None) -> None:
        self.publish(replace(self.view, state=state, error=error))
        self.loop.call_soon_threadsafe(self._fanout, None)

    def _deliver(self, current: JobView) -> None:
        self.writes.put_nowait(current)
        self._fanout(current)

    def _fanout(self, current: JobView | None) -> None:
        for queue in self.watchers:
            queue.put_nowait(current)

    @contextmanager
    def watch(self) -> Generator[asyncio.Queue[JobView | None]]:
        queue: asyncio.Queue[JobView | None] = asyncio.Queue()
        self.watchers.add(queue)
        try:
            yield queue
        finally:
            self.watchers.discard(queue)


Work = Callable[[Job], None]
"""The blocking body of a job. It reports through the job it is handed, and a report is
also where it learns it was cancelled."""


class Jobs:
    """The registry every slow thing goes through, and the threads it runs on.

    Its own pool, and never the loop's default one: a load's body blocks its thread on a
    coroutine that needs a thread of the loop's own to finish, which on one shared executor
    closes a cycle where no load ever completes.

    Past `_WORKERS` a job waits for a thread with its row already reading `running` — which
    is what every body beginning with a `report` is for: a cancellation that landed during
    that wait ends the job before it starts paying for anything.
    """

    def __init__(self) -> None:
        self.on_change: Callable[[], None] = _nothing
        """That the list of jobs moved — a job starting, reporting or ending."""
        self._live: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._threads = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="mlx_omnia-job")
        self._writes: asyncio.Queue[JobView] = asyncio.Queue()
        self._writer: asyncio.Task[None] | None = None

    async def open(self) -> int:
        """Start the single writer and reconcile whatever a killed process left behind."""
        self._writer = asyncio.create_task(self._drain())
        return await abandon()

    async def close(self) -> None:
        await self._writes.join()
        if self._writer is not None:
            self._writer.cancel()
        self._threads.shutdown(wait=False)

    def start(self, subject: Subject, work: Work) -> Job:
        """Call from the loop: the job captures it to hand frames back from the thread."""
        now = time.time()
        current = JobView(
            id=uuid.uuid4().hex,
            kind=kind_of(subject),
            subject=subject,
            state="pending",
            progress=Progress(),
            created_at=now,
            updated_at=now,
        )
        job = Job(
            view=current,
            loop=asyncio.get_running_loop(),
            writes=self._writes,
            announce=self.on_change,
        )
        self._live[current.id] = job
        self._writes.put_nowait(current)
        task = asyncio.create_task(self._run(job, work))
        # The loop holds a running task weakly: without a reference here a job can be
        # collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def live(self, job_id: str) -> Job | None:
        """The ones still running, which are the only ones a cancellation reaches."""
        return self._live.get(job_id)

    async def found(self, job_id: str) -> JobView:
        """A live job answers with the frame it is holding, and only what is not running is
        read back from the file: the rows are written by one drain behind the work, so a job
        just created is answered before its first row lands rather than as a 404.

        And a job that just *ended* is waited for rather than answered as a 404: it leaves
        `_live` the moment its body returns, with its last frames still on the write queue,
        so a body short enough to finish before the client's next poll — a PUT on a model
        that is already resident — would otherwise be a job the daemon handed out and then
        denied ever having.
        """
        running = self.live(job_id)
        if running is not None:
            return running.view
        await self._writes.join()
        current = await view(job_id)
        if current is None:
            raise NoSuchJob(job_id)
        return current

    async def cancel(self, job_id: str) -> JobView:
        """The job as it still is: the flag is set here, and the work turns it into
        `cancelled` when it reaches its next report."""
        current = await self.found(job_id)
        running = self.live(job_id)
        if running is None:
            raise JobFinished(job_id, current.state)
        running.cancel()
        return current

    async def _drain(self) -> None:
        while True:
            current = await self._writes.get()
            try:
                await write(current)
            finally:
                self._writes.task_done()

    async def _run(self, job: Job, work: Work) -> None:
        try:
            if job.cancelled.is_set():
                job.finish("cancelled", None)
                return
            job.publish(replace(job.view, state="running"))
            await job.loop.run_in_executor(self._threads, work, job)
        except Cancelled:
            job.finish("cancelled", None)
        except asyncio.CancelledError:
            # The loop is going away, which is not the work finishing. Without this the row
            # stays `running` for ever and the next process lists a job nobody runs.
            job.finish("cancelled", None)
            raise
        except Exception as error:
            job.finish("error", f"{type(error).__name__}: {error}")
        else:
            # Work that answers the flag by returning rather than raising ends cancelled too.
            job.finish("cancelled" if job.cancelled.is_set() else "ok", None)
        finally:
            del self._live[job.id]
            self.on_change()
