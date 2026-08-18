"""The job primitive over HTTP, driven by a toy job whose blocking work only moves when the
test says.

Every assertion below is about a job caught in a known step: the work waits for a permit
before each step and acknowledges it after, so nothing here depends on a sleep being long
enough. The two properties the primitive exists for — a late subscriber is told where the
job is, and a cancellation reaches inside the blocking step — are exactly the two a
timing-based test would report as flaky instead of as broken.

The stand — the toy, the real uvicorn server and the polling helpers — is `job_stand.py`.
What the registry does off the loop, with no server in front of it, is `test_jobs_registry.py`.
"""

import importlib

import httpx
import pytest

from .job_stand import (
    Stand,
    Toy,
    active_ids,
    frames,
    fresh_state,
    get,
    listing,
    stand,
    start,
    toy,
    wait_for,
)

__all__ = ["fresh_state", "stand", "toy"]
"""The fixtures this module runs on, imported rather than repeated."""


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
def test_a_silent_step_is_held_open_by_keep_alives(
    stand: Stand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same reason as the chat stream: a step that reports nothing for seconds — a shard
    being hashed, a model being loaded — must not look to the client like a dead connection.

    The interval is shortened to what the test can wait for; what is asserted is that the
    comment arrives at all."""
    routes = importlib.import_module("mlx_omnia.server.api.management.jobs")
    monkeypatch.setattr(routes, "KEEP_ALIVE_SECONDS", 0.1)
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
