"""The two config claims that need a real socket or a second process.

`restart` is only falsifiable against a listening server, and the api key is only clearable
under the host the daemon actually bound — so these are split out of `test_config.py`, which
answers everything else through one in-process client.
"""

import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .test_config import build, patch, setting


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def serving(app: FastAPI) -> Generator[str]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_patching_the_port_answers_restart_and_the_daemon_stays_on_the_port_it_bound() -> None:
    """`restart` against a real socket, which is the only place the word is falsifiable. A
    daemon that moved on the PATCH would drop every client already pointed at it — and the
    client that sent the PATCH would be the first.

    The port to move to is chosen with the server already listening, so the ephemeral bind
    that picks it cannot hand back the one the server is holding.
    """
    with serving(build()) as base_url:
        moved = free_port()
        response = httpx.patch(f"{base_url}/admin/config", json={"port": moved}, timeout=10)

        assert response.status_code == 200, response.text
        assert response.json()["port"] == {"value": moved, "effect": "restart", "note": None}
        assert httpx.get(f"{base_url}/admin/config", timeout=10).status_code == 200
        with pytest.raises(httpx.ConnectError):
            httpx.get(f"http://127.0.0.1:{moved}/admin/config", timeout=10)


def test_what_a_patch_wrote_is_what_the_next_process_reads() -> None:
    """A restart, in a test: a second app over the same file, sharing nothing with the first.
    The `None` is half the point — stored as its own string it comes back as `"None"`, which
    is a TTL of five characters."""
    with TestClient(build()) as first:
        patch(first, idle_ttl_seconds=None, api_key="sk-local", not_resident="fail")

    with TestClient(build()) as second:
        # The key that was written is in force from the next request on, this one included.
        answer = second.get("/admin/config", headers={"Authorization": "Bearer sk-local"})
        assert answer.status_code == 200, answer.text
        body = answer.json()
        assert isinstance(body, dict)

    assert setting(body, "idle_ttl_seconds")["value"] is None
    assert setting(body, "api_key")["value"] == "sk-local"
    assert setting(body, "not_resident")["value"] == "fail"


def test_the_key_cannot_be_cleared_out_from_under_a_daemon_on_the_network() -> None:
    """`auth.check_bind` refuses to *come up* off the loopback without a key, and that is
    only the boot. Clearing it afterwards would leave every route but `/admin/health` open on
    the network — the delete of a checkpoint among them — with no restart to notice it at.

    On the loopback the same PATCH is fine: nothing outside the machine can reach it, which
    is the whole of why the key is optional there."""
    signed = {"Authorization": "Bearer sk-open"}
    with TestClient(build(host="0.0.0.0")) as exposed:
        set_key = exposed.patch("/admin/config", json={"api_key": "sk-open"})
        assert set_key.status_code == 200, set_key.text

        cleared = exposed.patch("/admin/config", json={"api_key": None}, headers=signed)

        assert cleared.status_code == 400
        assert "api_key" in cleared.json()["detail"] and "0.0.0.0" in cleared.json()["detail"]
        after = exposed.get("/admin/config", headers=signed).json()
        assert setting(after, "api_key")["value"] == "sk-open", "and nothing was written"

    with TestClient(build()) as local:
        clearing = local.patch("/admin/config", json={"api_key": None}, headers=signed)
        assert clearing.status_code == 200, clearing.text
