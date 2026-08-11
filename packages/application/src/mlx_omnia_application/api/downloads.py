"""Every download the daemon is running, keyed by the repository it is fetching.

It lives above the views because a download outlives the screen that started it: the tile
in Library and the ring in the title bar read the same frames. The boot pass is what makes
a restart work — the active jobs say which downloads are running, and each one's
`subject.model` says which tile it belongs under.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import flet as ft

from mlx_omnia_application.api import engine as engine_api
from mlx_omnia_application.api.engine import Job

# Smoothed, because a shard already in the staging cache completes in a single frame and
# the instantaneous reading there is gigabytes per second.
SMOOTHING = 0.35


@dataclass
class Pull:
    job: str
    repo: str
    state: str
    message: str
    completed: int
    # Absent until the first response: a bar drawn from a guessed total is worse than none.
    total: int | None
    error: str | None
    # Bytes per second, null until a second frame gives it something to divide.
    rate: float | None
    at: float

    @property
    def fraction(self) -> float | None:
        return None if not self.total else self.completed / self.total

    @property
    def share(self) -> str:
        part = self.fraction
        return "—" if part is None else f"{round(part * 100)}%"

    @property
    def eta(self) -> float | None:
        if self.total is None or self.rate is None or self.rate <= 0:
            return None
        return (self.total - self.completed) / self.rate


@ft.observable
class Downloads:
    """Every download in flight, folded from the streams that report them.

    Observable: Library reads `active`, so a progress frame landing redraws the shelf and
    nothing else — and `active` is a dict, which means writing a key is the notification.
    """

    def __init__(self) -> None:
        self.active: dict[str, Pull] = {}
        self._followed: set[str] = set()

    def follow(self, job: Job) -> None:
        """Watch one download's stream until it ends. Idempotent per job, so the boot pass
        and a fresh `pull` naming the same job start one watcher between them."""
        if job["kind"] != "download" or job["id"] in self._followed:
            return
        self._followed.add(job["id"])
        repo = job["subject"]["model"]
        if not isinstance(repo, str):
            return
        self._fold(repo, job)
        asyncio.get_running_loop().create_task(self._watch(job["id"], repo))

    async def _watch(self, job: str, repo: str) -> None:
        try:
            async for frame in engine_api.job_events(job):
                self._fold(repo, frame)
        except Exception:  # noqa: BLE001
            # The stream dying is not itself news: a daemon that went away is already
            # drawn by the engine chip.
            pass
        finally:
            self._followed.discard(job)
            self.notify()

    def _fold(self, repo: str, frame: Job) -> None:
        previous = self.active.get(repo)
        # A finished download is the catalog's now, and a cancelled one took its bytes
        # with it; only a failure has something left to offer, so only a failure stays.
        if frame["state"] in ("ok", "cancelled"):
            self.active.pop(repo, None)
            return
        self.active[repo] = Pull(
            job=frame["id"],
            repo=repo,
            state=frame["state"],
            message=frame["progress"]["message"],
            completed=frame["progress"]["completed"],
            total=frame["progress"]["total"],
            error=frame["error"],
            rate=_rate(previous, frame),
            at=frame["updated_at"],
        )

    def forget(self, repo: str) -> None:
        """A terminal failure stays on screen until it is acted on; this is the acting."""
        self.active.pop(repo, None)

    async def boot(self) -> None:
        """What was already running before this window opened."""
        try:
            for job in await engine_api.active_jobs():
                self.follow(job)
        except Exception:  # noqa: BLE001 — a daemon that is down has nothing running
            pass


def _rate(previous: Pull | None, frame: Job) -> float | None:
    if previous is None:
        return None
    seconds = frame["updated_at"] - previous.at
    if seconds <= 0:
        return previous.rate
    instant = (frame["progress"]["completed"] - previous.completed) / seconds
    if previous.rate is None:
        return instant
    return previous.rate + SMOOTHING * (instant - previous.rate)
