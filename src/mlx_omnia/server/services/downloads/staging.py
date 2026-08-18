from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.downloads.hub import slug
from mlx_omnia.server.services.jobs import Jobs
from mlx_omnia.server.services.quantize import STAGING as QUANTIZE_STAGING

STAGING = ".incomplete"

DOWNLOADING: dict[str, str] = {}
"""The last job started for each repository, which is how a second request finds the first."""

DROPPING: set[str] = set()
"""Repositories whose bytes are being collected. Held across the `rmtree` and read by
`start_download` on the same loop, so a download cannot start fetching into a directory that
is being walked away."""


class AlreadyDownloading(Exception):
    """A live job is already fetching the repository. Carries its id."""

    def __init__(self, repository: str, job_id: str) -> None:
        super().__init__(f"{repository!r} is already downloading as job {job_id}")
        self.repository = repository
        self.job_id = job_id


class BeingCollected(Exception):
    """The repository's bytes are being deleted right now."""


class AlreadyOnDisk(Exception):
    """The hub cache already holds the folder, complete or not."""


class NothingStaged(Exception):
    """No staging area answers for the repository."""


@dataclass(frozen=True)
class Staged:
    """A repository with bytes on disk and no model to show for them: the staging cache of a
    failed download, or a `models--*` in the cache proper the catalog hides for being
    incomplete. The second has nowhere else to go — a download is refused because the directory
    is there, and the catalog cannot delete what its scan does not list."""

    repo: str
    bytes_on_disk: int


def running(registry: Jobs, repository: str) -> str | None:
    """The job downloading it, if one still is. The registry has the last word rather than the
    dict: a job cancelled before its work ever ran leaves its entry behind, and a repository
    that could never be downloaded again is a worse leak than a stale key."""
    job_id = DOWNLOADING.get(repository)
    return None if job_id is None or registry.live(job_id) is None else job_id


def _bytes_on_disk(directory: Path) -> int:
    """Only real files: a snapshot is symlinks into `blobs/`, and adding both sides reports
    twice the disk."""
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _abandoned() -> list[Path]:
    """Every directory `Staged` is about, the two staging areas first. The catalog decides that
    a `models--*` in the cache proper is abandoned: being listed is exactly what it means for a
    repository to have a model to show for its bytes.

    A quantization's area is here rather than in a window of its own because what a caller wants
    is what is taking space and can be dropped, and a run killed halfway leaves tens of
    gigabytes under a dot-directory neither the catalog nor `hf cache scan` shows."""
    roots = (catalog.HUB_CACHE / STAGING, catalog.HUB_CACHE / QUANTIZE_STAGING)
    staged = [path for root in roots if root.is_dir() for path in sorted(root.glob("models--*"))]
    if not catalog.HUB_CACHE.is_dir():
        return staged
    listed = {entry.store for entry in catalog.scan()}
    return staged + [
        path for path in sorted(catalog.HUB_CACHE.glob("models--*")) if path not in listed
    ]


def staged_repositories() -> list[Staged]:
    """The bytes a download left behind, and what they cost. One row per repository even when
    it has bytes in both places, because the repository is the unit a delete removes."""
    totals: dict[str, int] = {}
    for path in _abandoned():
        repo = path.name.removeprefix("models--").replace("--", "/")
        totals[repo] = totals.get(repo, 0) + _bytes_on_disk(path)
    return [Staged(repo=repo, bytes_on_disk=total) for repo, total in totals.items()]


async def drop_staged(registry: Jobs, repo: str) -> None:
    """Giving up on the resume. Refused while a job is fetching into it: `rmtree` under a
    running `hf_hub_download` is a download that dies on a directory that stopped existing, and
    cancelling the job is how that one is stopped.

    On the loop up to the claim, and only then in a thread: run whole in the threadpool, the
    refusal above stops meaning anything, since a download could register between the check and
    the `rmtree` walking the directory it just started fetching into.
    """
    if (job_id := running(registry, repo)) is not None:
        raise AlreadyDownloading(repo, job_id)
    folder = slug(repo)
    directories = [path for path in _abandoned() if path.name == folder]
    if not directories:
        raise NothingStaged(f"nothing is staged for {repo!r}")
    DROPPING.add(repo)
    try:
        for directory in directories:
            await asyncio.to_thread(shutil.rmtree, directory, True)
    finally:
        DROPPING.discard(repo)
