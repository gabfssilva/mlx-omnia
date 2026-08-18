"""The stand `test_jobs.py` drives the job primitive from: a toy whose blocking work only
moves when the test says, and a real server in front of it.

The server is a real one, unlike the rest of the `/admin` suites: `TestClient` runs the whole
ASGI response to completion before handing it over, and a stream that only ends when the job
does would deadlock the thread that has to feed the job.

The state directory is wiped once for the module rather than once per test: the database is
one file now, and the server holding it open outlives every test in it.
"""

import asyncio
import json
import queue
import shutil
import socket
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from mlx_omnia import paths
from mlx_omnia.server.api.management.common import accepted
from mlx_omnia.server.api.management.jobs import router
from mlx_omnia.server.db import base as db
from mlx_omnia.server.main import migrate
from mlx_omnia.server.services.jobs import Job, Jobs, Load, Progress


@pytest.fixture(scope="module", autouse=True)
def fresh_state() -> None:
    """The harness's per-test wipe, once for the module: the server below keeps the file open
    across every test in it."""
    root = paths.state_dir()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


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
    toy: Toy | None = None
    """Rebound per test: one server, and the work of whichever test is running."""


@pytest.fixture(scope="module")
def stand() -> Iterator[Stand]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    here = Stand(base_url=f"http://127.0.0.1:{port}")
    registry = Jobs()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await asyncio.to_thread(migrate)
        await db.connect()
        await registry.open()
        yield
        await registry.close()
        await db.disconnect()

    app = FastAPI(lifespan=lifespan)
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
