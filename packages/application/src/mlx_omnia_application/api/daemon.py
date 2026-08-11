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

from mlx_omnia_application.api.http import BASE

# packages/application/src/mlx_omnia_application/daemon.py — the checkout is five up, and
# it is where `uv run` has to be pointed when the console script is not on this path.
ROOT = pathlib.Path(os.environ.get("OMNIA_PROJECT") or pathlib.Path(__file__).parents[4])

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


async def _spawn() -> asyncio.subprocess.Process:
    beside = pathlib.Path(sys.executable).parent / "omnia-server"
    command = (
        [str(beside)]
        if beside.exists()
        else ["uv", "run", "--project", str(ROOT), "omnia-server"]
    )
    # The daemon's half of "quit is a stop": it watches this pid and shuts itself down
    # when it goes, which is the only path that survives a force quit.
    process = await asyncio.create_subprocess_exec(
        *command, "--parent-pid", str(os.getpid()), cwd=ROOT
    )
    # Quit is a stop, immediately. The --parent-pid watchdog is what covers the paths
    # this never runs on.
    atexit.register(_term, process.pid)
    return process
