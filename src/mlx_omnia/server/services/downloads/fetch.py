from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub

from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.downloads.hub import HUB, missing, slug, variant, wanted
from mlx_omnia.server.services.downloads.staging import (
    DOWNLOADING,
    DROPPING,
    STAGING,
    AlreadyDownloading,
    AlreadyOnDisk,
    BeingCollected,
    running,
)
from mlx_omnia.server.services.jobs import Cancelled, Download, Job, Jobs, Progress, Work

_REPORT_SECONDS = 0.25


class _Bytes:
    """The one place bytes are counted, and the one place a cancellation lands inside a 4 GB
    shard: every chunk reads the flag, which costs nothing, and only every `_REPORT_SECONDS`
    does it publish — a report is a row written to the database and a frame to every watcher."""

    def __init__(self, job: Job, total: int) -> None:
        self._job = job
        self._total = total
        self._published = 0.0
        self.done = 0
        self.file = ""

    def add(self, size: float) -> None:
        self.done += int(size)
        self.publish()

    def publish(self, force: bool = False) -> None:
        if self._job.cancelled.is_set():
            raise Cancelled(self._job.id)
        now = time.monotonic()
        if not force and now - self._published < _REPORT_SECONDS:
            return
        self._published = now
        self._job.report(Progress(message=self.file, completed=self.done, total=self._total))


class _Bar:
    """What `hf_hub_download` builds to draw a progress bar with; here it only forwards bytes.
    One per file, with tqdm's keyword arguments, of which `initial` — what a previous attempt
    left on disk — is the one that means something. A server that ignores a `Range` header
    hands back a negative `update`, and the counter follows."""

    def __init__(self, counter: _Bytes, *, initial: float = 0, **_: object) -> None:
        self._counter = counter
        counter.add(initial)

    def __enter__(self) -> _Bar:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, n: float | None = 1) -> None:
        self._counter.add(n or 0)

    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        """The xet path writes a transfer rate into the bar it was handed; we draw none."""


if TYPE_CHECKING:
    # `hf_hub_download` types `tqdm_class` as a tqdm subclass, and its own progress-bar factory
    # documents the opposite: a callable that is not a `tqdm` type is called with the bar's
    # keyword arguments and used as it is. The corrected signature is here, the way
    # `engine/core/mxcompat.py` carries mlx's; at runtime the name binds to the real function.

    def hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        revision: str,
        cache_dir: Path,
        tqdm_class: Callable[..., _Bar],
    ) -> str: ...

else:
    hf_hub_download = huggingface_hub.hf_hub_download


def _install(staged: Path, sha: str, final: Path) -> None:
    """The rename that gives the snapshot its final name, and what has to hold first. The
    reference is written here rather than left to the resolver: the downloads are pinned to the
    sha, and `refs/main` is how the catalog knows which revision to read."""
    absent = missing(staged / "snapshots" / sha)
    if absent:
        raise FileNotFoundError(f"{final.name} is missing {', '.join(absent)}")
    reference = staged / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(sha)
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        staged.rename(final)
    except OSError as error:
        raise RuntimeError(
            f"{final.name} appeared in the hub cache while it was being downloaded;"
            f" what was fetched is staged at {staged}"
        ) from error


def _download(repository: str) -> Work:
    def work(job: Job) -> None:
        staging = catalog.HUB_CACHE / STAGING
        staged = staging / slug(repository)
        # Every exit of the work is inside: the ending below is a policy, so an ending it does
        # not cover is bytes abandoned. The rename is the worst of them — it is the one moment
        # the staging holds the whole repository.
        try:
            # A report first: a job cancelled while it waited for a thread must not open a
            # connection to the Hub, let alone start paying for the bytes.
            job.report(Progress(message=f"listing {repository}"))
            info = HUB.model_info(repository, files_metadata=True)
            sha = info.sha
            if sha is None:
                raise ValueError(f"{repository!r} names no revision to pin the download to")
            files = wanted(info)
            counter = _Bytes(job, sum(size for _, size in files))
            for index, (name, size) in enumerate(files, start=1):
                counter.file = f"{name} ({index}/{len(files)})"
                counter.publish(force=True)
                base = counter.done
                hf_hub_download(
                    repository,
                    name,
                    revision=sha,
                    cache_dir=staging,
                    tqdm_class=partial(_Bar, counter),
                )
                # A file already in the staging cache reports nothing at all, and a resumed one
                # reports less than its size: what the file was worth is known here.
                counter.done = max(counter.done, base + size)
            counter.publish(force=True)
            _install(staged, sha, catalog.HUB_CACHE / slug(repository))
        except Exception as error:
            # A cancellation raised inside the xet callback crosses a Rust frame on its way
            # out, and what arrives here may be any exception carrying the message; the flag is
            # what says which ending this is, not the type.
            if not job.cancelled.is_set():
                raise
            # Whoever cancelled asked not to have this repository, so its staging goes with the
            # job — including whatever an earlier failed attempt had banked there.
            shutil.rmtree(staged, ignore_errors=True)
            raise Cancelled(job.id) from error

    return work


async def start_download(registry: Jobs, repo: str, quant: str | None = None) -> Job:
    """On the loop, because `start` captures it to hand frames back from the worker thread —
    which is also what makes the checks below a gate: a dict lookup and one `stat`, with no
    `await` between reading them and registering the job that answers them."""
    repository = variant(repo, quant)
    if (job_id := running(registry, repository)) is not None:
        raise AlreadyDownloading(repository, job_id)
    if repository in DROPPING:
        raise BeingCollected(f"{repository!r} is being collected")
    # The folder rather than the catalog: an interrupted download leaves a `models--*` the scan
    # hides for being incomplete, and letting it through means paying for the whole repository
    # and only then failing on a rename into a directory that is already there.
    if (catalog.HUB_CACHE / slug(repository)).exists():
        raise AlreadyOnDisk(f"{repository!r} is already on disk")
    job = registry.start(Download(model=repository), _download(repository))
    DOWNLOADING[repository] = job.id
    return job
