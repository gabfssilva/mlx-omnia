"""/admin/metrics/events: the stream that publishes the register.

Watching a generation is not taking part in it — a client that subscribes mid-generation is
told where the request is, and one that leaves neither cancels nor slows it.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from mlx_omnia.server.api.management.common import KEEP_ALIVE_SECONDS
from tests.server.metrics_stand import (
    PACE,
    PACED_TOKENS,
    TINY_ACTIVE_BYTES,
    Stand,
    entries,
    entry,
    frames,
    number,
    run,
    snapshot,
    stand,
    wait_for_a_token,
)

__all__ = ["stand"]


def test_a_client_that_subscribes_mid_generation_is_told_where_the_request_is(
    stand: Stand,
) -> None:
    """The whole reason the stream opens with the current state. A first frame that carried
    only the next transition would tell a dashboard connecting during a long generation
    nothing at all until it ended — which is when its numbers stop being live.

    The clock is part of the assertion: the silent tick resamples every 0.5s, so a stream
    that answered from the tick instead of from the subscription would still show a live
    request — half a second late, and only because something else was already moving."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(run, stand, "paced", "hello", PACED_TOKENS)
        started = wait_for_a_token(stand)
        asked = time.monotonic()
        with (
            httpx.Client() as http,
            http.stream("GET", f"{stand.base_url}/admin/metrics/events", timeout=30) as response,
        ):
            assert response.status_code == 200
            first = next(frames(response))
            waited = time.monotonic() - asked
        finished = running.result(timeout=60)

    assert waited < KEEP_ALIVE_SECONDS / 2, (
        "the first frame waited for a tick instead of carrying the state"
    )
    live = first["live"]
    assert isinstance(live, list) and len(live) == 1, "the first frame carried no running request"
    record = entry(live[0])
    assert record["model"] == "paced"
    assert record["state"] == "running"
    assert record["bytes_per_token"] == TINY_ACTIVE_BYTES
    assert number(record, "ttft") > 0
    tokens = number(record, "completion_tokens")
    assert number(started, "completion_tokens") <= tokens < PACED_TOKENS
    assert finished["completion_tokens"] == PACED_TOKENS


_BURST = 3
"""Requests fired back to back, well inside one keep-alive tick. Each moves the register
twice — one starts, one ends — so the burst owes the stream more frames than resampling
could account for."""


def counted(frame: dict[str, object], model: str) -> float:
    for record in entries(frame, "models"):
        if record["model"] == model:
            return number(record, "requests")
    return 0.0


def test_the_stream_keeps_publishing_after_the_frame_the_subscription_owes(
    stand: Stand,
) -> None:
    """Every other test here reads one frame and stops, which leaves the loop *after* the
    subscription untested: replacing its body with a bare keep-alive, or dropping the fanout
    `_publish` writes to the watchers, keeps the suite green over a GET wearing an SSE costume.

    The clock is what separates publishing from resampling. Once the burst is over nothing
    moves again, so a stream that only resampled on the silent tick would find an unchanged
    snapshot and emit keep-alives for ever; under a budget of `_KEEP_ALIVE_SECONDS` per pair of
    frames it cannot deliver them even while the burst is still running.
    """
    budget = _BURST * KEEP_ALIVE_SECONDS
    with (
        httpx.Client() as http,
        http.stream("GET", f"{stand.base_url}/admin/metrics/events", timeout=30) as response,
    ):
        assert response.status_code == 200
        stream = frames(response, budget)
        before = counted(next(stream), "quick")
        for _ in range(_BURST):
            run(stand, "quick", "hello", 2)
        published = [next(stream) for _ in range(_BURST * 2)]

    assert counted(published[-1], "quick") > before, "every frame carried the opening snapshot"


def test_closing_the_stream_neither_cancels_nor_slows_the_generation(stand: Stand) -> None:
    """The chat's own SSE ends in `job.cancel()` — a client that walks away stops paying for
    tokens. This one must not: watching a generation is not taking part in it, and a `finally`
    that cancelled here would let one dashboard truncate another client's answer.

    The rate is the evidence for the second half, and only the floor of it: a decode thread
    that had to wait on the closed stream falls far under `1/PACE`, while exceeding it is
    impossible by construction — the paced model sleeps `PACE` per step, so a ceiling here
    would be an assertion about the double and not about the register.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(run, stand, "paced", "hello", PACED_TOKENS)
        wait_for_a_token(stand)
        with (
            httpx.Client() as http,
            http.stream("GET", f"{stand.base_url}/admin/metrics/events", timeout=30) as response,
        ):
            assert next(frames(response))["live"] != []
        finished = running.result(timeout=60)

    assert finished["state"] == "completed"
    assert finished["completion_tokens"] == PACED_TOKENS
    record = entries(snapshot(stand), "requests")[0]
    assert record["state"] == "completed"
    assert record["completion_tokens"] == PACED_TOKENS
    assert number(record, "tokens_per_second") > 0.25 / PACE

    deadline = time.monotonic() + 10
    while stand.metrics.watchers:
        assert time.monotonic() < deadline, "the closed stream left its queue subscribed"
