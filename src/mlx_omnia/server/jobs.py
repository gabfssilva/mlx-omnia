"""A job is a state plus a stream of events, and the stream is the dialects' own SSE.

Every frame is the whole state, never a delta, and the stream opens with the state as it is
now: progress that only exists in the events is a privilege of whoever connected first. The
subscription is registered *before* that first state is read, so a transition landing in
between is delivered twice instead of being lost — a repeated frame costs nothing when the
frame is the whole state.

Cancellation is cooperative and has to reach inside the blocking work. The job body runs in
a thread of this module's own pool (see `Jobs`), where an `asyncio.Task.cancel()` interrupts
neither an MLX load nor a socket read: `DELETE` sets a `threading.Event` and `report` — the
same call the work already makes to publish progress — raises `Cancelled` when it finds it
set. That is the one way out of a download callback that has survived contact with
production, and it is worth inheriting.

The row in `store.jobs` is written on every frame, so `GET` is answered from the file: a job
that outlived the process still reports where it stopped. What stays in memory is only what
cannot be persisted — the cancellation flag and the open streams.
"""

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omnia.server.store import JobRecord, JobState, Store

_KEEP_ALIVE_SECONDS = 0.5

_WORKERS = 4
"""How many job bodies run at once. The gate that serializes generation is the engine's and
is depth one, so nothing here is a number about the GPU: what these threads do is disk and
network, where a handful of concurrent streams already has the device and more of them only
splits the same bandwidth into more pieces that each finish later."""

_TERMINAL: frozenset[JobState] = frozenset({"ok", "error", "cancelled"})


class Cancelled(Exception):
    """Raised inside the worker thread, by `Job.report`, when the job was cancelled."""


@dataclass(frozen=True)
class Progress:
    """What every kind reports the same way. `total` is absent while it is unknown — a
    download learns its size from the first response, and a bar drawn from a guessed total
    is worse than no bar."""

    message: str = ""
    completed: float = 0.0
    total: float | None = None


@dataclass(frozen=True)
class Download:
    """The Hub repository the bytes are coming from."""

    model: str


@dataclass(frozen=True)
class Load:
    model: str


@dataclass(frozen=True)
class Quantize:
    model: str
    target: str


@dataclass(frozen=True)
class Bench:
    model: str


@dataclass(frozen=True)
class Benchmark:
    """A batch, which is about several checkpoints at once — the one subject that is not one
    model. `kind` is which of the three views it fills."""

    kind: str
    models: list[str]


Subject = Download | Load | Quantize | Bench | Benchmark
"""What the job is about, which is the one thing `kind` alone never said. A screen that draws
a job beside the thing it acts on — the download under its checkpoint — needs the pairing to
survive a reload, and `progress.message` is a sentence for a person, not a key."""


def _kind(subject: Subject) -> str:
    match subject:
        case Download():
            return "download"
        case Load():
            return "load"
        case Quantize():
            return "quantize"
        case Bench():
            return "bench"
        case Benchmark():
            return "benchmark"


def _subject(kind: str, text: str) -> Subject:
    """The column is JSON text for the same reason `progress` is: a table under it would make
    every field a kind grows a migration of the other three."""
    match (kind, json.loads(text)):
        case ("download", {"model": str(model)}):
            return Download(model=model)
        case ("load", {"model": str(model)}):
            return Load(model=model)
        case ("quantize", {"model": str(model), "target": str(target)}):
            return Quantize(model=model, target=target)
        case ("bench", {"model": str(model)}):
            return Bench(model=model)
        case ("benchmark", {"kind": str(kind), "models": list(models)}) if all(
            isinstance(name, str) for name in models
        ):
            return Benchmark(kind=kind, models=[name for name in models if isinstance(name, str)])
        case other:
            raise ValueError(f"malformed job subject: {other!r}")


@dataclass(frozen=True)
class JobView:
    """One shape for the four windows on a job: the body of the `202`, of the `GET`, of the
    listing, and of every SSE frame. `kind` is the tag `subject` is read back with."""

    id: str
    kind: str
    subject: Subject
    state: JobState
    progress: Progress
    created_at: float
    updated_at: float
    error: str | None = None


def _record(view: JobView) -> JobRecord:
    return JobRecord(
        id=view.id,
        kind=view.kind,
        subject=json.dumps(asdict(view.subject)),
        state=view.state,
        progress=json.dumps(asdict(view.progress)),
        created_at=view.created_at,
        updated_at=view.updated_at,
        error=view.error,
    )


def _progress(text: str) -> Progress:
    """The column is JSON text, so the way back is a parse and not a cast: a row written by
    an older version fails here, named, instead of reaching a client as a half-filled bar."""
    match json.loads(text):
        case {
            "message": str(message),
            "completed": (int() | float()) as completed,
            "total": (None | int() | float()) as total,
        }:
            return Progress(message=message, completed=completed, total=total)
        case other:
            raise ValueError(f"malformed job progress: {other!r}")


def _view(record: JobRecord) -> JobView:
    return JobView(
        id=record.id,
        kind=record.kind,
        subject=_subject(record.kind, record.subject),
        state=record.state,
        progress=_progress(record.progress),
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=record.error,
    )


def _nothing() -> None:
    pass


@dataclass
class Job:
    """The half of a job that cannot be a row: the flag the blocking work reads and the
    streams waiting on it."""

    view: JobView
    loop: asyncio.AbstractEventLoop
    store: Store
    cancelled: threading.Event = field(default_factory=threading.Event)
    watchers: set[asyncio.Queue[JobView | None]] = field(default_factory=set)
    announce: Callable[[], None] = _nothing
    """That the list of jobs moved — the registry's `on_change`, handed down by `start`.
    Called from whichever thread published, so whatever is behind it has to take one."""

    @property
    def id(self) -> str:
        return self.view.id

    def cancel(self) -> None:
        self.cancelled.set()

    def report(self, progress: Progress) -> None:
        """The work's only door, and the reason it is also where cancellation is found: a
        flag read between steps never interrupts the step itself, and the work already calls
        this often enough to report."""
        if self.cancelled.is_set():
            raise Cancelled(self.view.id)
        self.publish(replace(self.view, state="running", progress=progress))

    def publish(self, view: JobView) -> None:
        """The row goes through whichever thread is calling — `Store` opens a connection per
        call, so there is no object crossing threads — and the fanout is handed to the loop
        that owns the queues.

        Mostly the worker thread, and not only: the loop itself publishes the creation, the
        move to `running` and the terminal frame. Those three are a sqlite write on the loop,
        which is worth knowing before writing a fourth — under a busy WAL the wait is the
        store's `timeout`, and what is on the loop with it is a stream of tokens."""
        self.view = replace(view, updated_at=time.time())
        self.store.save_job(_record(self.view))
        self.loop.call_soon_threadsafe(self._fanout, self.view)
        self.announce()

    def finish(self, state: JobState, error: str | None) -> None:
        """The terminal frame and then the end of every stream, in that order."""
        self.publish(replace(self.view, state=state, error=error))
        self.loop.call_soon_threadsafe(self._fanout, None)

    def _fanout(self, view: JobView | None) -> None:
        for queue in self.watchers:
            queue.put_nowait(view)

    @contextmanager
    def watch(self) -> Generator[asyncio.Queue[JobView | None]]:
        queue: asyncio.Queue[JobView | None] = asyncio.Queue()
        self.watchers.add(queue)
        try:
            yield queue
        finally:
            self.watchers.discard(queue)


Work = Callable[[Job], None]
"""The blocking body of a job. It reports through the job it is handed, and a report is also
where it learns it was cancelled — so beginning with one is what answers a cancellation that
arrived while the call was still waiting for a thread in the pool."""


class Jobs:
    """The registry every slow thing goes through, and the threads it runs on.

    Its own pool, and never the loop's default one. A job body blocks its thread until it is
    done, and a load's body blocks it on a coroutine that needs a thread of the loop's own to
    finish (`residency.py`). On one shared executor that closes a cycle: `min(32, cpu + 4)`
    concurrent loads hold every thread `Engine._load`, `Engine._decode` and admission's
    `_checkpoint_size` are waiting for, so no load finishes, no generation starts, and Ctrl-C
    does not end the process — the interpreter joins those threads on the way out. Separated,
    nothing a job body waits for is waiting on a job body.

    Past `_WORKERS` a job waits for a thread with its row already reading `running` — which is
    what every body beginning with a `report` is for: a cancellation that landed during that
    wait ends the job before it starts paying for anything.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self.on_change: Callable[[], None] = _nothing
        """Raised whenever the list `GET /admin/jobs` answers with moves — a job starting,
        reporting or ending. What listens is the event stream."""
        self._live: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._threads = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="mlx_omnia-job")
        # Whatever the last process was running when it was killed. Only a kill leaves one —
        # the loop's own shutdown finishes what it has — and nothing else would ever move it:
        # the screen would open on a bar that never fills, and the cancel beside it answers
        # 409, because the id belongs to no live job here.
        self._store.abandon_jobs()

    def start(self, subject: Subject, work: Work) -> Job:
        """Call from the loop: the job captures it to hand frames back from the thread."""
        now = time.time()
        view = JobView(
            id=uuid.uuid4().hex,
            kind=_kind(subject),
            subject=subject,
            state="pending",
            progress=Progress(),
            created_at=now,
            updated_at=now,
        )
        job = Job(
            view=view,
            loop=asyncio.get_running_loop(),
            store=self._store,
            announce=self.on_change,
        )
        self._live[view.id] = job
        self._store.save_job(_record(view))
        task = asyncio.create_task(self._run(job, work))
        # The loop holds a running task weakly: without a reference here a job can be
        # collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def live(self, job_id: str) -> Job | None:
        """The ones still running, which are the only ones a cancellation reaches."""
        return self._live.get(job_id)

    def view(self, job_id: str) -> JobView | None:
        return None if (record := self._store.job(job_id)) is None else _view(record)

    def recent(self) -> list[JobView]:
        return [_view(record) for record in self._store.jobs()]

    async def _run(self, job: Job, work: Work) -> None:
        try:
            if job.cancelled.is_set():
                # Cancelled between `start` and the loop reaching this task. The engine checks
                # the same flag in the same place (`engine.py`) and for the same reason: what
                # the work begins with is a download, and not beginning it is the point.
                job.finish("cancelled", None)
                return
            job.publish(replace(job.view, state="running"))
            await job.loop.run_in_executor(self._threads, work, job)
        except Cancelled:
            job.finish("cancelled", None)
        except asyncio.CancelledError:
            # The loop is going away, which is not the same as the work finishing. Without
            # this the row stays `running` for ever: the next process reads it from the file
            # and lists a job nobody is running, and `DELETE` answers 409.
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


def _registry(request: Request) -> Jobs:
    registry = request.app.state.jobs
    assert isinstance(registry, Jobs)
    return registry


JobsDep = Annotated[Jobs, Depends(_registry)]

router = APIRouter()


def accepted(job: Job) -> JSONResponse:
    """What every creator answers. The first frame travels in the body so the client knows
    the id and the state without a second round trip to the `Location` it was just given."""
    return JSONResponse(
        status_code=202,
        content=asdict(job.view),
        headers={"Location": f"/admin/jobs/{job.id}"},
    )


def _found(registry: Jobs, job_id: str) -> JobView:
    view = registry.view(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    return view


def _frame(view: JobView) -> str:
    return f"data: {json.dumps(asdict(view))}\n\n"


async def _last(view: JobView) -> AsyncIterator[str]:
    """A job nobody runs any more: its state is the whole stream."""
    yield _frame(view)


async def _frames(job: Job) -> AsyncIterator[str]:
    """Subscribed by `watch` before the state is read, and unsubscribed on the way out —
    including when the client walks away mid-job, which unlike the chat stream cancels
    nothing: a job outlives whoever was watching it."""
    with job.watch() as queue:
        current = job.view
        yield _frame(current)
        if current.state in _TERMINAL:
            return
        while True:
            try:
                view = await asyncio.wait_for(queue.get(), _KEEP_ALIVE_SECONDS)
            except TimeoutError:
                # Keeps the connection warm through a long, silent step.
                yield ": keep-alive\n\n"
                continue
            if view is None:
                return
            yield _frame(view)


@router.get("/admin/jobs")
def jobs(registry: JobsDep, active: bool = False) -> list[JobView]:
    """Newest first, as the store orders them. `active=true` is the app's progress list; the
    unfiltered one is the history, which is what a client that missed a job reads."""
    return [view for view in registry.recent() if not active or view.state not in _TERMINAL]


@router.get("/admin/jobs/{job_id}")
def job(job_id: str, registry: JobsDep) -> JobView:
    return _found(registry, job_id)


@router.delete("/admin/jobs/{job_id}", status_code=202)
def cancel(job_id: str, registry: JobsDep) -> JobView:
    """202 and the job as it still is: the flag is set here, and the work turns it into
    `cancelled` when it reaches its next report. Announcing `cancelled` from this handler
    would be announcing something that has not happened yet."""
    view = _found(registry, job_id)
    running = registry.live(job_id)
    if running is None:
        raise HTTPException(
            status_code=409, detail=f"job {job_id!r} already finished as {view.state}"
        )
    running.cancel()
    return view


@router.get("/admin/jobs/{job_id}/events")
async def events(job_id: str, registry: JobsDep) -> StreamingResponse:
    running = registry.live(job_id)
    stream = _last(_found(registry, job_id)) if running is None else _frames(running)
    return StreamingResponse(stream, media_type="text/event-stream")
