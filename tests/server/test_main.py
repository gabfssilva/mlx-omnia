"""`--parent-pid`: the daemon's own half of "killing the app kills the server".

The shell that spawns the daemon already stops it on quit, but only along paths where it
gets to run code. A force quit, a crash or a SIGKILL leaves the daemon holding the port
with a model resident — and the next start finds a daemon it did not spawn and refuses to
stop it. So the guarantee is tested from the daemon's side: a real server, a real owner,
and the owner killed the way nothing can be forwarded through.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from mlx_omnia.server.main import watch_parent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def owner() -> Iterator[subprocess.Popen[bytes]]:
    """A process to be the daemon's owner, and nothing else. Killed by the test that owns
    it; the terminate here is for the tests that leave it alive on purpose."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield process
    finally:
        process.kill()
        process.wait()


def test_the_watch_fires_once_the_owner_is_gone(owner: subprocess.Popen[bytes]) -> None:
    fired = threading.Event()
    watch_parent(owner.pid, fired.set, every=0.02)

    owner.kill()
    owner.wait()

    assert fired.wait(timeout=5), "the owner died and the watch did not fire"


def test_the_watch_stays_quiet_while_the_owner_lives(owner: subprocess.Popen[bytes]) -> None:
    """The other half of the claim, and the one that would make the flag unusable if it were
    wrong: a daemon that shuts down on a living owner is a daemon that shuts down at random.
    """
    fired = threading.Event()
    watch_parent(owner.pid, fired.set, every=0.02)

    assert not fired.wait(timeout=1), "the watch fired on a process that is still running"


def test_a_daemon_started_with_parent_pid_stops_when_that_pid_is_killed(
    owner: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    """The whole path, through the console script's own argument parsing: a real daemon
    answering on a real port, its owner SIGKILLed so nothing can be forwarded to it, and the
    daemon gone on its own. `OMNIA_STATE_DIR` so the boot does not touch the user's db.
    """
    port = free_port()
    daemon = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_omnia.server.main",
            "--port",
            str(port),
            "--parent-pid",
            str(owner.pid),
        ],
        env={**os.environ, "OMNIA_STATE_DIR": str(tmp_path)},
    )
    try:
        health = f"http://127.0.0.1:{port}/admin/health"
        deadline = time.time() + 120
        while True:
            assert daemon.poll() is None, "the daemon exited before it answered"
            assert time.time() < deadline, "the daemon never came up"
            try:
                httpx.get(health, timeout=1).raise_for_status()
                break
            except (httpx.HTTPError, OSError):
                time.sleep(0.2)

        owner.kill()
        owner.wait()

        try:
            daemon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail("the daemon outlived the owner it was told to watch")
    finally:
        if daemon.poll() is None:
            daemon.kill()
            daemon.wait()
