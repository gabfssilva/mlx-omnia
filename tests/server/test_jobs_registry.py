"""The registry off the loop: what a job costs before it starts, which threads the bodies get,
and what is left in the file when the process that ran them is gone.

Nothing here goes through HTTP — these are the properties no request can put itself in the
middle of: a cancellation that lands before the first step, a pool that is this registry's own,
and a row written by a registry that no longer exists. The database is the one
`mlx_omnia.paths` names, which the harness gives every test fresh; each test opens and closes
its own connection, so two `asyncio.run` calls in one test are two processes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager

from mlx_omnia.server.db import base as db
from mlx_omnia.server.main import migrate
from mlx_omnia.server.services.jobs import (
    Download,
    Job,
    Jobs,
    JobView,
    Load,
    Progress,
    recent,
    view,
)
from mlx_omnia.server.services.jobs.registry import _WORKERS
from mlx_omnia.server.services.jobs.views import write


def in_database[T](body: Callable[[], Coroutine[None, None, T]]) -> T:
    """One process's worth of database: the migration, the connection, and the loop the
    registry captured — all gone by the time the call returns."""
    migrate()

    async def main() -> T:
        await db.connect()
        try:
            return await body()
        finally:
            await db.disconnect()

    return asyncio.run(main())


@asynccontextmanager
async def running() -> AsyncGenerator[Jobs]:
    registry = Jobs()
    await registry.open()
    try:
        yield registry
    finally:
        await registry.close()


def test_a_job_cancelled_before_its_turn_never_enters_the_work() -> None:
    """Nothing else can put the cancellation before the task's first step. A `DELETE` that
    lands while the job is still `pending` must not let the work begin — for a download that
    is the difference between cancelling it and paying for it in full."""
    entered = threading.Event()

    def work(job: Job) -> None:
        entered.set()
        job.report(Progress(message="started"))

    async def run() -> str:
        async with running() as registry:
            job = registry.start(Load(model="toy"), work)
            job.cancel()
            deadline = time.monotonic() + 5
            while registry.live(job.id) is not None:
                assert time.monotonic() < deadline, "the job never left the live set"
                await asyncio.sleep(0.01)
            current = await registry.found(job.id)
            return current.state

    assert in_database(run) == "cancelled"
    assert not entered.is_set(), "the work started for a job that was already cancelled"


def test_the_bodies_run_on_this_registry_s_own_pool_and_not_more_than_it_is_wide() -> None:
    """The half a shared executor fails outright: the bodies run on threads this registry
    owns, so a client cannot turn `_WORKERS + 2` PUTs into `_WORKERS + 2` threads of the
    loop's — which is where `Engine._load` and `Engine._generate` have to find one.

    The peak is what says it, and no window is needed to read it in: on the loop's own
    executor all `_WORKERS + 2` bodies start in the same turn and the peak is `_WORKERS + 2`,
    where a pool of `_WORKERS` threads cannot run more than `_WORKERS` of them whatever the
    timing. The two that got no thread wait with their row already reading `running`, which
    is the second assertion — and the reason every body begins with a `report`.
    """
    entered = 0
    active = 0
    peak = 0
    counting = threading.Lock()
    release = threading.Event()

    def work(_job: Job) -> None:
        nonlocal entered, active, peak
        with counting:
            entered += 1
            active += 1
            peak = max(peak, active)
        assert release.wait(5), "the test never released the bodies"
        with counting:
            active -= 1

    async def run() -> tuple[int, list[str]]:
        async with running() as registry:
            started = [registry.start(Load(model="toy"), work) for _ in range(_WORKERS + 2)]
            deadline = time.monotonic() + 5
            try:
                while True:
                    with counting:
                        if entered >= _WORKERS:
                            break
                    assert time.monotonic() < deadline, f"only {entered} bodies took a thread"
                    await asyncio.sleep(0.01)
                waiting: list[str] = [job.view.state for job in started]
            finally:
                release.set()
            drained = time.monotonic() + 5
            while any(registry.live(job.id) is not None for job in started):
                assert time.monotonic() < drained, "a job never left the live set"
                await asyncio.sleep(0.01)
            finished = [(await registry.found(job.id)).state for job in started]
            assert finished == ["ok"] * len(started), finished
            return peak, waiting

    held, waiting = in_database(run)

    assert held == _WORKERS, "the bodies did not run on a pool of this registry's own"
    assert waiting == ["running"] * (_WORKERS + 2), "a job waiting for a thread reads as pending"


def test_a_job_whose_loop_goes_away_does_not_stay_running() -> None:
    """Shutting the daemon down cancels the task that is awaiting the worker thread. The row
    is the state, so a job left `running` in the file is a job the next process lists as
    active for ever and refuses to cancel."""
    release = threading.Event()

    def work(_job: Job) -> None:
        release.wait(5)

    async def run() -> str:
        async with running() as registry:
            before = asyncio.all_tasks()
            job = registry.start(Load(model="toy"), work)
            deadline = time.monotonic() + 5
            while True:
                if job.view.state == "running":
                    break
                assert time.monotonic() < deadline, "the job never reached the worker thread"
                await asyncio.sleep(0.01)
            for task in asyncio.all_tasks() - before:
                task.cancel()
            release.set()
            await asyncio.sleep(0.05)
            return job.id

    job_id = in_database(run)

    async def reopened() -> JobView | None:
        return await view(job_id)

    found = in_database(reopened)
    assert found is not None
    assert found.state == "cancelled"


def test_the_row_outlives_the_registry_that_wrote_it() -> None:
    """The state is the row, so a restarted daemon — a second `Jobs` over the same file —
    still says where the job stopped, progress included."""

    def work(job: Job) -> None:
        job.report(Progress(message="step 1", completed=1.0, total=1.0))

    async def run() -> str:
        async with running() as registry:
            job = registry.start(Load(model="toy"), work)
            deadline = time.monotonic() + 5
            while registry.live(job.id) is not None:
                assert time.monotonic() < deadline, "the job never finished"
                await asyncio.sleep(0.01)
            return job.id

    job_id = in_database(run)

    async def reopened() -> tuple[JobView | None, list[str]]:
        async with running():
            return await view(job_id), [stored.id for stored in await recent()]

    found, history = in_database(reopened)
    assert found is not None
    assert found.state == "ok"
    assert found.progress == Progress(message="step 1", completed=1.0, total=1.0)
    assert job_id in history


def test_a_job_the_last_process_left_running_is_not_running_here() -> None:
    """A `kill -9` mid-download is the ordinary end of a daemon, and the row it leaves says
    `running` for ever: the screen opens on a bar that never fills and the cancel beside it
    answers 409, because no live job here carries that id. The registry reconciles what it
    finds when it comes up, and says why in the row itself."""

    async def killed() -> None:
        await write(
            JobView(
                id="abandoned",
                kind="download",
                subject=Download(model="mlx-community/qwen3-30b"),
                state="running",
                progress=Progress(message="half a shard"),
                created_at=time.time(),
                updated_at=time.time(),
            )
        )

    in_database(killed)

    async def reopened() -> JobView | None:
        async with running():
            return await view("abandoned")

    found = in_database(reopened)
    assert found is not None
    assert found.state == "error"
    assert found.error is not None and "daemon stopped" in found.error
    assert found.progress.message == "half a shard", "where it stopped, kept"
