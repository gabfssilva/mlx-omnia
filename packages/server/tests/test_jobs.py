"""The job primitive, driven by a toy job whose blocking work only moves when the test says.

Every assertion below is about a job caught in a known step: the work waits for a permit
before each step and acknowledges it after, so nothing here depends on a sleep being long
enough. The two properties the primitive exists for — a late subscriber is told where the
job is, and a cancellation reaches inside the blocking step — are exactly the two a
timing-based test would report as flaky instead of as broken.

The server is a real one, unlike the rest of the `/admin` suites: `TestClient` runs the
whole ASGI response to completion before handing it over, and a stream that only ends when
the job does would deadlock the thread that has to feed the job.
"""

import asyncio
import json
import queue
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from mlx_omnia_server.jobs import _WORKERS, Job, Jobs, Load, Progress, accepted, router
from mlx_omnia_server.store import JobRecord, Store


@dataclass
class Toy:
    """Blocking work with a gate and a side effect: the marks file is the real work a
    cancellation has to stop. Reporting comes before the mark, because the report is where
    the cancellation is found — a step that was cancelled must leave no mark behind."""

    marks: Path
    steps: int = 3
    fail: str | None = None
    permits: queue.Queue[None] = field(default_factory=queue.Queue)
    acknowledged: queue.Queue[int] = field(default_factory=queue.Queue)

    def __call__(self, job: Job) -> None:
        for step in range(1, self.steps + 1):
            self.permits.get()
            job.report(Progress(message=f"step {step}", completed=step, total=self.steps))
            with self.marks.open("a") as file:
                file.write(f"{step}\n")
            self.acknowledged.put(step)
        if self.fail is not None:
            raise RuntimeError(self.fail)

    def step(self) -> int:
        """One permit, and the step it lands on once the work has taken it."""
        self.permits.put(None)
        return self.acknowledged.get(timeout=5)

    def release(self) -> None:
        """Permits for whatever is still waiting: a step left blocked holds a pool thread,
        and the interpreter joins those at exit — the suite would hang there, not here."""
        for _ in range(self.steps + 1):
            self.permits.put(None)


@dataclass
class Stand:
    base_url: str
    database: Path
    toy: Toy | None = None
    """Rebound per test: one server, and the work of whichever test is running."""


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    directory = tmp_path_factory.mktemp("jobs")
    here = Stand(base_url=f"http://127.0.0.1:{port}", database=directory / "server.db")
    registry = Jobs(Store(here.database))

    app = FastAPI()
    app.include_router(router)
    app.state.jobs = registry

    async def start_toy() -> JSONResponse:
        """The creator's whole side of the contract, which is what `accepted` is for."""
        assert here.toy is not None
        return accepted(registry.start(Load(model="toy"), here.toy))

    app.post("/toy")(start_toy)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield here
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def toy(stand: Stand, tmp_path: Path) -> Iterator[Toy]:
    work = Toy(marks=tmp_path / "marks")
    work.marks.write_text("")
    stand.toy = work
    try:
        yield work
    finally:
        work.release()
        settle(stand)


def settle(stand: Stand) -> None:
    """The next test's `active` list must be about the next test. A step left waiting would
    also hold a pool thread the interpreter joins at exit."""
    deadline = time.monotonic() + 5
    while active_ids(stand) and time.monotonic() < deadline:
        time.sleep(0.01)


def start(stand: Stand) -> str:
    response = httpx.post(f"{stand.base_url}/toy")
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    assert isinstance(job_id, str)
    return job_id


def get(stand: Stand, job_id: str) -> dict[str, object]:
    response = httpx.get(f"{stand.base_url}/admin/jobs/{job_id}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def listing(stand: Stand, active: bool = False) -> list[dict[str, object]]:
    query = {"active": "true"} if active else {}
    response = httpx.get(f"{stand.base_url}/admin/jobs", params=query)
    assert response.status_code == 200, response.text
    return response.json()


def active_ids(stand: Stand) -> list[object]:
    return [view["id"] for view in listing(stand, active=True)]


def wait_for(stand: Stand, job_id: str, state: str) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while True:
        view = get(stand, job_id)
        if view["state"] == state:
            return view
        assert time.monotonic() < deadline, f"job stayed in {view['state']!r}, wanted {state!r}"
        time.sleep(0.01)


def frames(response: httpx.Response, seconds: float = 30.0) -> Iterator[dict[str, object]]:
    """Bounded by the clock, not by the transport: the keep-alive this module emits every
    0.5s keeps httpx's read timeout from ever firing, so a fanout that regressed would hang
    the whole suite instead of failing this test."""
    deadline = time.monotonic() + seconds
    for line in response.iter_lines():
        assert time.monotonic() < deadline, "the stream never said what the test waits for"
        if line.startswith("data: "):
            payload = json.loads(line.removeprefix("data: "))
            assert isinstance(payload, dict)
            yield payload


@pytest.mark.usefixtures("toy")
def test_creating_a_job_answers_202_with_its_location(stand: Stand) -> None:
    """The shape every creator repeats — download, quantize, bench and residency — which is
    why it lives in the primitive instead of being written four times."""
    response = httpx.post(f"{stand.base_url}/toy")

    assert response.status_code == 202, response.text
    body = response.json()
    assert response.headers["location"] == f"/admin/jobs/{body['id']}"
    assert body["kind"] == "load"
    assert body["subject"] == {"model": "toy"}
    assert body["state"] == "pending"
    assert httpx.get(f"{stand.base_url}{response.headers['location']}").status_code == 200


def test_a_job_reports_progress_finishes_and_leaves_the_active_list(stand: Stand, toy: Toy) -> None:
    job_id = start(stand)

    assert toy.step() == 1
    running = get(stand, job_id)
    assert running["state"] == "running"
    assert running["progress"] == {"message": "step 1", "completed": 1, "total": 3}
    assert job_id in active_ids(stand)

    toy.step()
    toy.step()
    finished = wait_for(stand, job_id, "ok")

    assert finished["error"] is None
    assert finished["progress"] == {"message": "step 3", "completed": 3, "total": 3}
    assert toy.marks.read_text() == "1\n2\n3\n"
    assert job_id not in active_ids(stand)
    assert job_id in [view["id"] for view in listing(stand)]


def test_a_client_that_subscribes_after_the_start_is_told_where_the_job_is(
    stand: Stand, toy: Toy
) -> None:
    """The first frame is the current state, not the next transition — otherwise progress
    belongs to whoever connected first. A frame already delivered may repeat (the
    subscription is registered before the state is read, on purpose), so what the second
    assertion waits for is the first frame that says something new."""
    job_id = start(stand)
    assert toy.step() == 1

    with (
        httpx.Client() as http,
        http.stream("GET", f"{stand.base_url}/admin/jobs/{job_id}/events") as response,
    ):
        assert response.status_code == 200
        stream = frames(response)

        first = next(stream)
        assert first["state"] == "running"
        assert first["progress"] == {"message": "step 1", "completed": 1, "total": 3}

        toy.step()
        following = next(frame for frame in stream if frame["progress"] != first["progress"])
        assert following["progress"] == {"message": "step 2", "completed": 2, "total": 3}

        toy.step()
        assert [frame["state"] for frame in stream][-1] == "ok"


@pytest.mark.usefixtures("toy")
def test_a_silent_step_is_held_open_by_keep_alives(stand: Stand) -> None:
    """The same reason as the chat stream (`app.py`): a step that reports nothing for
    seconds — a shard being hashed, a model being loaded — must not look to the client like
    a dead connection."""
    job_id = start(stand)

    with (
        httpx.Client() as http,
        http.stream("GET", f"{stand.base_url}/admin/jobs/{job_id}/events") as response,
    ):
        # No permit was handed out, so the only frames a running job can still emit are the
        # state it opened with and its transition to `running`: six lines is past both.
        opening = [line for _, line in zip(range(6), response.iter_lines(), strict=False)]

    assert any(line.startswith("data: ") for line in opening)
    assert any(line.startswith(":") for line in opening)


def test_subscribing_to_a_job_that_already_finished_is_one_frame_and_the_end(
    stand: Stand, toy: Toy
) -> None:
    """The extreme of the same rule, and the one that would hang: a stream waiting for
    events a finished job will never send."""
    toy.steps = 1
    job_id = start(stand)
    toy.step()
    wait_for(stand, job_id, "ok")

    with (
        httpx.Client() as http,
        http.stream("GET", f"{stand.base_url}/admin/jobs/{job_id}/events") as response,
    ):
        collected = list(frames(response))

    assert [frame["state"] for frame in collected] == ["ok"]
    assert collected[0]["progress"] == {"message": "step 1", "completed": 1, "total": 1}


def test_a_delete_stops_the_work_and_the_job_ends_cancelled(stand: Stand, toy: Toy) -> None:
    """`asyncio.Task.cancel()` would leave the thread inside the step it is running. What
    stops it is the flag the work reads on its next report — and the mark that report was
    about is the side effect that must never happen."""
    job_id = start(stand)
    assert toy.step() == 1

    response = httpx.delete(f"{stand.base_url}/admin/jobs/{job_id}")
    assert response.status_code == 202, response.text
    assert response.json()["state"] == "running"

    toy.permits.put(None)
    cancelled = wait_for(stand, job_id, "cancelled")

    assert cancelled["error"] is None
    assert cancelled["progress"] == {"message": "step 1", "completed": 1, "total": 3}
    assert toy.marks.read_text() == "1\n"
    assert toy.acknowledged.empty()
    assert job_id not in active_ids(stand)


def test_cancelling_a_job_that_already_finished_is_refused(stand: Stand, toy: Toy) -> None:
    toy.steps = 1
    job_id = start(stand)
    toy.step()
    wait_for(stand, job_id, "ok")

    response = httpx.delete(f"{stand.base_url}/admin/jobs/{job_id}")

    assert response.status_code == 409, response.text
    assert "ok" in response.json()["detail"]
    assert get(stand, job_id)["state"] == "ok"


def test_a_job_that_fails_keeps_the_message(stand: Stand, toy: Toy) -> None:
    toy.steps = 1
    toy.fail = "no space left on device"
    job_id = start(stand)
    toy.step()

    failed = wait_for(stand, job_id, "error")

    assert failed["error"] == "RuntimeError: no space left on device"
    assert failed["progress"] == {"message": "step 1", "completed": 1, "total": 1}
    assert job_id not in active_ids(stand)


def test_an_unknown_job_is_a_404_on_every_route(stand: Stand) -> None:
    assert httpx.get(f"{stand.base_url}/admin/jobs/nope").status_code == 404
    assert httpx.get(f"{stand.base_url}/admin/jobs/nope/events").status_code == 404
    assert httpx.delete(f"{stand.base_url}/admin/jobs/nope").status_code == 404


def test_a_job_cancelled_before_its_turn_never_enters_the_work(tmp_path: Path) -> None:
    """Driven off the loop rather than over HTTP: nothing else can put the cancellation
    before the task's first step. A `DELETE` that lands while the job is still `pending`
    must not let the work begin — for a download that is the difference between cancelling
    it and paying for it in full."""
    entered = threading.Event()

    def work(job: Job) -> None:
        entered.set()
        job.report(Progress(message="started"))

    async def run() -> str:
        registry = Jobs(Store(tmp_path / "server.db"))
        job = registry.start(Load(model="toy"), work)
        job.cancel()
        deadline = time.monotonic() + 5
        while registry.live(job.id) is not None:
            assert time.monotonic() < deadline, "the job never left the live set"
            await asyncio.sleep(0.01)
        view = registry.view(job.id)
        assert view is not None
        return view.state

    assert asyncio.run(run()) == "cancelled"
    assert not entered.is_set(), "the work started for a job that was already cancelled"


def test_the_bodies_run_on_this_registry_s_own_pool_and_not_more_than_it_is_wide(
    tmp_path: Path,
) -> None:
    """The other half of task 48, and the one a shared executor fails outright: the bodies run
    on threads this registry owns, so a client cannot turn `_WORKERS + 2` PUTs into
    `_WORKERS + 2` threads of the loop's — which is where `Engine._load` and `Engine._decode`
    have to find one.

    The peak is what says it, and no window is needed to read it in: on the loop's own
    executor all `_WORKERS + 2` bodies start in the same turn and the peak is `_WORKERS + 2`,
    where a pool of `_WORKERS` threads cannot run more than `_WORKERS` of them whatever the
    timing. The two that got no thread wait with their row already reading `running`, which
    is the second assertion — and the reason every body begins with a `report`.
    """
    entered = 0
    running = 0
    peak = 0
    counting = threading.Lock()
    release = threading.Event()

    def work(_job: Job) -> None:
        nonlocal entered, running, peak
        with counting:
            entered += 1
            running += 1
            peak = max(peak, running)
        assert release.wait(5), "the test never released the bodies"
        with counting:
            running -= 1

    async def run() -> tuple[int, list[str]]:
        registry = Jobs(Store(tmp_path / "server.db"))
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
        finished: list[str] = [
            view.state for job in started if (view := registry.view(job.id)) is not None
        ]
        assert finished == ["ok"] * len(started), finished
        return peak, waiting

    held, waiting = asyncio.run(run())

    assert held == _WORKERS, "the bodies did not run on a pool of this registry's own"
    assert waiting == ["running"] * (_WORKERS + 2), "a job waiting for a thread reads as pending"


def test_a_job_whose_loop_goes_away_does_not_stay_running(tmp_path: Path) -> None:
    """Shutting the daemon down cancels the task that is awaiting the worker thread. The row
    is the state, so a job left `running` in the file is a job the next process lists as
    active for ever and refuses to cancel."""
    database = tmp_path / "server.db"
    release = threading.Event()

    def work(_job: Job) -> None:
        release.wait(5)

    async def run() -> str:
        registry = Jobs(Store(database))
        job = registry.start(Load(model="toy"), work)
        deadline = time.monotonic() + 5
        while True:
            view = registry.view(job.id)
            assert view is not None
            if view.state == "running":
                break
            assert time.monotonic() < deadline, "the job never reached the worker thread"
            await asyncio.sleep(0.01)
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current:
                task.cancel()
        release.set()
        await asyncio.sleep(0.05)
        return job.id

    job_id = asyncio.run(run())

    view = Jobs(Store(database)).view(job_id)
    assert view is not None
    assert view.state == "cancelled"


def test_the_row_outlives_the_registry_that_wrote_it(stand: Stand, toy: Toy) -> None:
    """The state is the row, so a restarted daemon — a second `Jobs` over the same file —
    still says where the job stopped, progress included."""
    toy.steps = 1
    job_id = start(stand)
    toy.step()
    wait_for(stand, job_id, "ok")

    reopened = Jobs(Store(stand.database))

    view = reopened.view(job_id)
    assert view is not None
    assert view.state == "ok"
    assert view.progress == Progress(message="step 1", completed=1.0, total=1.0)
    assert job_id in [stored.id for stored in reopened.recent()]


def test_a_job_the_last_process_left_running_is_not_running_here(tmp_path: Path) -> None:
    """A `kill -9` mid-download is the ordinary end of a daemon, and the row it leaves says
    `running` for ever: the screen opens on a bar that never fills and the cancel beside it
    answers 409, because no live job here carries that id. The registry reconciles what it
    finds when it comes up, and says why in the row itself."""
    path = tmp_path / "server.db"
    killed = Store(path)
    killed.save_job(
        JobRecord(
            id="abandoned",
            kind="download",
            subject=json.dumps({"model": "mlx-community/qwen3-30b"}),
            state="running",
            progress=json.dumps({"message": "half a shard"}),
            created_at=time.time(),
            updated_at=time.time(),
        )
    )

    Jobs(Store(path))

    found = Store(path).job("abandoned")
    assert found is not None
    assert found.state == "error"
    assert found.error is not None and "daemon stopped" in found.error
    assert json.loads(found.progress)["message"] == "half a shard", "where it stopped, kept"
