"""The daemon this CLI starts when nothing is answering at the address.

Three decisions live here.

**Where the log goes.** The child is detached and nobody is watching its terminal, so its
output goes to `${XDG_CONFIG_HOME:-~/.config}/sideros/daemon.log` — the same file the app's
Settings card already names, because it is the same daemon: one address, one port, one log.
It is truncated at each start; uvicorn writes an access line per request, and the run whose
lines answer "why is it not up" is the one that has just failed.

**One daemon, not two.** Two CLIs starting from zero both find `/admin/health` down. What
serializes them is a `flock` on `daemon.lock`, taken without blocking: the winner spawns and
the loser — who never tries again — waits for the winner's health, which is the same wait it
would be doing anyway. The port would serialize them too, at the cost of a second
interpreter that imports mlx only to die on the bind. Health is re-read inside the lock
before spawning, because a lock acquired is not evidence that the last holder failed.

**Detached from this process's signals.** `start_new_session=True`: without it the daemon
sits in the CLI's process group and the Ctrl-C that is meant to cancel one generation kills
the engine it was running on. Nothing here stops the child on the way out either — a daemon
that dies with the client that happened to start it is not a daemon.
"""

import fcntl
import os
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from sideros_cli.client import PROBE_TIMEOUT, ServerError, health

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

BOOT_TIMEOUT = 90.0
"""It boots with nothing resident, so this is an mlx import and a bind, not a load."""

_POLL_SECONDS = 0.25


def state_dir() -> Path:
    """Where the app writes `app.json` and the server its `server.db`, same XDG rule."""
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root) if root else Path.home() / ".config"
    return base / "sideros"


def log_path() -> Path:
    return state_dir() / "daemon.log"


def lock_path() -> Path:
    return state_dir() / "daemon.lock"


def address(base_url: str) -> tuple[str, int] | None:
    """Host and port when the URL names a daemon this machine can start. Any other host
    belongs to someone else's daemon and this process is merely its client."""
    parts = urlsplit(base_url)
    if parts.hostname not in LOOPBACK:
        return None
    return parts.hostname or "127.0.0.1", parts.port or 8642


def live(base_url: str, *, timeout: float = PROBE_TIMEOUT) -> bool:
    try:
        health(base_url, timeout=timeout)
    except ServerError:
        return False
    return True


def reason(log: str) -> str:
    """Why the daemon stopped, out of what it wrote. It runs under uvicorn, whose shutdown
    chatter is the last thing in the file and would bury the line that matters."""
    lines = [line for line in log.splitlines() if line.strip() and not line.startswith("INFO:")]
    return " · ".join(lines[-3:])


def spawn(host: str, port: int, log: Path) -> subprocess.Popen[bytes]:
    """Runs the server as a child of this interpreter: the CLI cannot import it — that would
    put mlx in this process — and `sys.executable` is the one interpreter known to have it."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        return subprocess.Popen(
            [sys.executable, "-m", "sideros_server.main", "--host", host, "--port", str(port)],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


@contextmanager
def _claim(path: Path) -> Generator[bool]:
    """True to whoever holds the lock, False to whoever finds it held — never a wait."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _wait(base_url: str, child: subprocess.Popen[bytes] | None, log: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if live(base_url, timeout=1.0):
            return
        if child is not None and child.poll() is not None:
            # The child is gone. Either it lost the port to a daemon started outside this
            # lock — the app's — or it failed: the probe says which.
            if live(base_url, timeout=1.0):
                return
            raise ServerError(
                f"the daemon exited without answering at {base_url}: "
                f"{reason(_tail(log)) or f'nothing in {log}'}"
            )
        time.sleep(_POLL_SECONDS)
    raise ServerError(f"the daemon did not answer at {base_url} within {timeout:g}s; see {log}")


def _tail(log: Path) -> str:
    try:
        return log.read_text(errors="replace")
    except OSError:
        return ""


def start(base_url: str, *, timeout: float = BOOT_TIMEOUT) -> None:
    """Brings a daemon up at `base_url`, assuming the caller has already found none there."""
    where = address(base_url)
    if where is None:
        raise ServerError(f"{base_url} is not a loopback address: no daemon can be started here")
    host, port = where
    log = log_path()
    with _claim(lock_path()) as won:
        child = spawn(host, port, log) if won and not live(base_url) else None
        _wait(base_url, child, log, timeout)
