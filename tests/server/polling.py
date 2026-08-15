"""Reading an admin job while it runs: the three moves every job suite makes.

Every wait is bounded — a job that never arrives fails the suite instead of hanging it.
"""

import time
from collections.abc import Mapping

from fastapi.testclient import TestClient

DEADLINE = 30.0


def view(client: TestClient, job_id: str) -> dict[str, object]:
    response = client.get(f"/admin/jobs/{job_id}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def wait_for(
    client: TestClient, job_id: str, *states: str, seconds: float = DEADLINE
) -> dict[str, object]:
    deadline = time.monotonic() + seconds
    while True:
        current = view(client, job_id)
        if current["state"] in states:
            return current
        assert time.monotonic() < deadline, (
            f"job stayed in {current['state']!r} ({current['error']!r}), wanted {states!r}"
        )
        time.sleep(0.01)


def progress(current: Mapping[str, object]) -> tuple[float, float]:
    frame = current["progress"]
    assert isinstance(frame, dict)
    completed, total = frame["completed"], frame["total"]
    assert isinstance(completed, int | float) and isinstance(total, int | float)
    return completed, total
