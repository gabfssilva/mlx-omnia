"""The engine process, and whether this window is the one that started it.

The window discovers a daemon on /admin/health, starts one when nobody answers, and owns
only the one it started: quitting takes that daemon with it, and Settings' Stop/Restart
refuse on any other with the reason.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import pathlib
import signal
import sys

import flet as ft
import httpx

from mlx_omnia import paths
from mlx_omnia.app.api.http import BASE

# src/mlx_omnia/app/api/daemon.py — the checkout is four up, and it is where `uv run` has
# to be pointed when the console script is not on this path.
ROOT = pathlib.Path(os.environ.get("OMNIA_PROJECT") or pathlib.Path(__file__).parents[4])

def bundled_python() -> pathlib.Path | None:
    """The engine's own interpreter inside the .app, or None outside one.

    Flet embeds Python as a dylib in the Flutter host and ships no executable, so in the
    packaged window `sys.executable` is the window itself and there is nothing there to
    exec. `mise run dmg` lays a CPython beside it with the engine installed into it.

    Nothing absolute is stored, which is what lets the bundle be dragged into /Applications
    — the same reason the console scripts in `engine/bin` are unusable here, since their
    shebangs hold the path they were installed under.
    """
    app = paths.bundle()
    if app is None:
        return None
    python = app / "Contents" / "Resources" / "engine" / "bin" / "python3"
    return python if python.exists() else None

FOREIGN = "The engine was started outside the app. Stop it where it was started."


@ft.observable
class Daemon:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        # Outlives a failed start, so a daemon that never comes up is not respawned on
        # every probe.
        self.claimed = False

    @property
    def owned(self) -> bool:
        return self.process is not None

    async def boot(self) -> None:
        """Start one if nobody answers. Never blocks the window: the rail's dot is what
        reports a daemon that did not come up."""
        if self.claimed or await up():
            return
        self.claimed = True
        self.process = await _spawn()

    async def stop(self) -> None:
        """SIGTERM and wait: uvicorn shuts down and the port is free for a respawn. No
        SIGKILL fallback — a daemon that ignores SIGTERM is reported, not escalated."""
        child = self.process
        if child is None:
            raise RuntimeError(
                "the daemon was not started by this app; stop it where it was started"
            )
        self.process, self.claimed = None, False
        child.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(child.wait(), 10)
        except TimeoutError:
            self.process, self.claimed = child, True
            raise RuntimeError("the daemon did not exit within 10s of SIGTERM") from None

    async def restart(self) -> None:
        if self.owned:
            await self.stop()
        elif await up():
            raise RuntimeError(
                "the daemon was not started by this app; restart it where it was started"
            )
        self.claimed = True
        self.process = await _spawn()


def _term(pid: int) -> None:
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)


async def up() -> bool:
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=2) as client:
            await client.get("/admin/health")
    except httpx.HTTPError:
        return False
    return True


def _command() -> list[str]:
    """How to start the engine, in the three places the window runs from."""
    if (bundled := bundled_python()) is not None:
        return [str(bundled), "-m", "mlx_omnia.server.main"]
    beside = pathlib.Path(sys.executable).parent / "omnia-server"
    if beside.exists():
        return [str(beside)]
    return ["uv", "run", "--project", str(ROOT), "omnia-server"]


def child_environment() -> dict[str, str]:
    """The child's environment, with this interpreter's own settings taken out of it.

    serious_python points `PYTHONHOME` and `PYTHONPATH` at the window's embedded runtime,
    and a child inherits them: the engine's interpreter would then read the window's stdlib
    and the window's site-packages, where the server's dependencies are deliberately not.
    Inherited, it does not even reach an import error — `Py_Initialize` fails and the
    process dies with no Python frame to blame.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON") and key != "SERIOUS_PYTHON_APP"
    }


async def _spawn() -> asyncio.subprocess.Process:
    command = _command()
    bundled = bundled_python() is not None
    # The checkout is where `uv run` has to start; the bundle has no checkout, and `ROOT`
    # there is whatever lies four levels above a temp directory. The engine needs no working
    # directory of its own — the model cache is addressed absolutely through the hub.
    cwd = None if bundled else ROOT
    # The daemon's half of "quit is a stop": it watches this pid and shuts itself down
    # when it goes, which is the only path that survives a force quit.
    # Nobody is watching this child's terminal — in the bundle there is not one. Truncated
    # at each start, because the run whose lines answer "why is it not up" is the one that
    # has just failed, and uvicorn writes an access line per request.
    handle = paths.daemon_log().open("wb")
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            "--parent-pid",
            str(os.getpid()),
            cwd=cwd,
            env=child_environment() if bundled else None,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
        )
    finally:
        # The child holds its own descriptor now.
        handle.close()
    # Quit is a stop, immediately. The --parent-pid watchdog is what covers the paths
    # this never runs on.
    atexit.register(_term, process.pid)
    return process
