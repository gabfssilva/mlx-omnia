"""Downloading a repository is creating the model, so it is `POST /admin/models` and a job.

The bytes land in a staging cache — `<hub cache>/.incomplete/`, one level below what
`catalog.scan` globs — and the repository folder is renamed into the hub cache only once
every shard the `weight_map` names is there. A snapshot under its final name is a snapshot
the catalog offers and the loader opens; an interrupted download that gets that name is a
catalog entry that fails at request time, which is the failure the completeness rule exists
to avoid.

Staging is per repository and survives a failed attempt on purpose, and what it buys is
resume **per file**, not per byte. `hf_hub_download` (1.24) writes each file to a name
unique to the attempt and unlinks it on failure, so the shard that was interrupted starts
over; what a second attempt does not pay for again is every file that had already landed in
`blobs/` with its pointer. On a repository of a few large shards that is most of the bytes,
and it is the whole of the guarantee — a resume that picks up mid-shard is not something
this dependency offers.

Because the staging is per repository, two jobs on one repository are two jobs writing into
one directory: the first to finish renames it out from under the second, which then reports
a missing shard for bytes that did arrive. So a repository has at most one live download and
the second `POST` is refused with the id of the job already fetching it. Different
repositories stage in different folders and run side by side — a queue serializing them would
make the second wait for nothing.

Nothing sweeps the staging on a timer or at boot: from outside, forty gigabytes the user is
about to retry look exactly like forty gigabytes abandoned in March. What decides is how the
job ended. A **cancellation** is the one ending that says the repository is not wanted, and
it takes the staging with it; a **failure** keeps its bytes, which is what the next attempt
resumes from — and `GET /admin/hub/staging` lists them with what they cost, `DELETE` drops
them, so what is kept is never kept in silence.

`quant` names a variant already published on the Hub (`mlx-community/<name>-<quant>`) and
never a quantization run here: the published one was verified by whoever uploaded it and is
a fraction of the bytes, while quantizing at home costs the whole dense download plus the
load's memory. That is a different job, and it is 34.3's.
"""

import asyncio
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import httpx
import huggingface_hub
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import RepositoryNotFoundError
from pydantic import BaseModel, ConfigDict

from mlx_omnia.server import catalog, quantize
from mlx_omnia.server.jobs import (
    Cancelled,
    Download,
    Job,
    Jobs,
    JobsDep,
    Progress,
    Work,
    accepted,
)

_STAGING = ".incomplete"
_VARIANTS = "mlx-community"
_PATTERNS = ("*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer.model", "*.tiktoken")
_REPORT_SECONDS = 0.25

HUB = HfApi()

_DOWNLOADING: dict[str, str] = {}
"""The last job started for each repository, which is how a second `POST` finds the first."""


class _IndexJson(TypedDict):
    weight_map: dict[str, str]


class _Bytes:
    """The one place bytes are counted, and the one place a cancellation lands inside a 4 GB
    shard: every chunk reads the flag, which costs nothing, and only every `_REPORT_SECONDS`
    does it publish — a report is a row written to sqlite and a frame to every watcher, and
    one per 10 MB chunk would be a transaction per chunk."""

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
    """What `hf_hub_download` builds to draw a progress bar with; here it only forwards
    bytes. One is constructed per file, with tqdm's keyword arguments, of which `initial` —
    what a previous attempt left on disk — is the one that means something to us. A server
    that ignores a `Range` header hands back a negative `update`, and the counter follows."""

    def __init__(self, counter: _Bytes, *, initial: float = 0, **_: object) -> None:
        self._counter = counter
        counter.add(initial)

    def __enter__(self) -> "_Bar":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, n: float | None = 1) -> None:
        self._counter.add(n or 0)

    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        """The xet path writes a transfer rate into the bar it was handed; we draw none."""


if TYPE_CHECKING:
    # `hf_hub_download` types `tqdm_class` as a tqdm subclass, and its own progress-bar
    # factory documents the opposite: a callable that is not a `tqdm` type is called with
    # the bar's keyword arguments and used as it is (`utils/tqdm.py`, which guards the
    # `issubclass` against `functools.partial` by name). The corrected signature is here,
    # the same way `mlx_omnia/core/mxcompat.py` carries mlx's; at runtime the name binds to
    # the real function.

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


def _wanted(info: ModelInfo) -> list[tuple[str, int]]:
    """The file set the loader reads — config, tokenizer, index and shards — and nothing in
    a subfolder: `original/` in a Llama-shaped repository is a second copy of the weights in
    a format nothing here opens."""
    return [
        (sibling.rfilename, sibling.size or 0)
        for sibling in info.siblings or []
        if "/" not in sibling.rfilename
        and any(fnmatch(sibling.rfilename, pattern) for pattern in _PATTERNS)
    ]


def _running(registry: Jobs, repository: str) -> str | None:
    """The job downloading it, if one still is. The registry has the last word rather than
    the dict: a job cancelled before its work ever ran leaves its entry behind, and a
    repository that could never be downloaded again is a worse leak than a stale key."""
    job_id = _DOWNLOADING.get(repository)
    return None if job_id is None or registry.live(job_id) is None else job_id


def _missing(snapshot: Path) -> list[str]:
    """What the loader will ask for and did not arrive. The `weight_map` is the list
    `catalog` reads to decide a directory is a model, so it is the list that decides here
    too; `is_file` follows the symlink into `blobs/`. Nothing in the code holds this reading
    and `catalog._complete` together — a test does, because the day they disagree the
    download starts publishing what the catalog hides."""
    index = snapshot / "model.safetensors.index.json"
    names = ["config.json"]
    if index.is_file():
        weights: _IndexJson = json.loads(index.read_text())
        names += sorted(set(weights["weight_map"].values()))
    else:
        names.append("model.safetensors")
    return [name for name in names if not (snapshot / name).is_file()]


def _slug(repository: str) -> str:
    """The folder the hub cache gives a repository, which is also what a half-finished
    download leaves behind under a name `catalog.scan` refuses to list."""
    return f"models--{repository.replace('/', '--')}"


def _install(staged: Path, sha: str, final: Path) -> None:
    """The rename that gives the snapshot its final name, and what has to hold first. The
    reference is written here rather than left to the resolver: the downloads are pinned to
    the sha, and `refs/main` is how the catalog knows which revision to read."""
    absent = _missing(staged / "snapshots" / sha)
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
        staging = catalog.HUB_CACHE / _STAGING
        staged = staging / _slug(repository)
        # Every exit of the work is inside, and that is the point: the collection below is a
        # policy about the ending, so an ending it does not cover is bytes abandoned. The
        # listing and the rename are two of them, and the rename is the worst — it is the one
        # moment the staging holds the whole repository.
        try:
            # A report first: a job cancelled while it waited for a thread must not open a
            # connection to the Hub, let alone start paying for the bytes.
            job.report(Progress(message=f"listing {repository}"))
            info = HUB.model_info(repository, files_metadata=True)
            sha = info.sha
            if sha is None:
                raise ValueError(f"{repository!r} names no revision to pin the download to")
            files = _wanted(info)
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
                # A file already in the staging cache reports nothing at all, and a resumed
                # one reports less than its size: what the file was worth is known here.
                counter.done = max(counter.done, base + size)
            counter.publish(force=True)
            _install(staged, sha, catalog.HUB_CACHE / _slug(repository))
        except Exception as error:
            # A cancellation raised inside the xet callback crosses a Rust frame on its way
            # out, and what arrives here may be any exception carrying the message; the flag
            # is what says which ending this is, not the type.
            if not job.cancelled.is_set():
                raise
            # Whoever cancelled asked not to have this repository, so its staging goes with
            # the job — including whatever an earlier failed attempt had banked there. A
            # failure keeps its bytes; this is the ending that does not.
            shutil.rmtree(staged, ignore_errors=True)
            raise Cancelled(job.id) from error

    return work


@dataclass(frozen=True)
class HubModel:
    """What the download screen's search field lists."""

    id: str
    downloads: int | None
    likes: int | None


@dataclass(frozen=True)
class Staged:
    """A repository with bytes on disk and no model to show for them: the staging cache of a
    download that failed, or a `models--*` in the cache proper that the catalog hides for
    being incomplete. The second is the one with nowhere else to go — `create` refuses it
    because the directory is there and `catalog.remove` cannot reach it because the scan does
    not list it, so without this door it is a repository that can never be downloaded again."""

    repo: str
    bytes_on_disk: int


_DROPPING: set[str] = set()
"""Repositories whose bytes are being collected. Held across the `rmtree` and read by
`create` on the same loop, so a download cannot start fetching into a directory that is being
walked away — the check and the claim have no `await` between them, which is the same
property that makes `create`'s own gate a gate."""


def _bytes_on_disk(directory: Path) -> int:
    """Only real files: a snapshot is symlinks into `blobs/`, and adding both sides reports
    twice the disk."""
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _abandoned() -> list[Path]:
    """Every directory `Staged` is about, the two staging areas first. The catalog is what
    decides a `models--*` in the cache proper is abandoned: being listed is exactly what it
    means for a repository to have a model to show for its bytes.

    `quantize`'s area is here rather than in a window of its own because what a caller wants
    is what is taking space and can be dropped, and the two answer that the same way. A
    quantization killed halfway leaves tens of gigabytes under a dot-directory that neither
    the catalog nor `hf cache scan` shows — the same dead end the download's own staging was
    in before it was listed."""
    roots = (catalog.HUB_CACHE / _STAGING, catalog.HUB_CACHE / quantize.STAGING)
    staged = [path for root in roots if root.is_dir() for path in sorted(root.glob("models--*"))]
    if not catalog.HUB_CACHE.is_dir():
        return staged
    listed = {entry.store for entry in catalog.scan()}
    return staged + [
        path for path in sorted(catalog.HUB_CACHE.glob("models--*")) if path not in listed
    ]


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    quant: str | None = None


def _variant(repo: str, quant: str | None) -> str:
    """Where `quant` sends the download: the variant mlx-community publishes under that
    name. Nothing is quantized here."""
    return repo if quant is None else f"{_VARIANTS}/{repo.rsplit('/', 1)[-1]}-{quant}"


router = APIRouter()


@router.post("/admin/models", status_code=202)
async def create(request: DownloadRequest, registry: JobsDep) -> JSONResponse:
    """On the loop, because `start` captures it to hand frames back from the worker thread —
    which is also what makes the two checks below a gate: a dict lookup and one `stat`, with
    no `await` between reading them and registering the job that answers them."""
    repository = _variant(request.repo, request.quant)
    if (job_id := _running(registry, repository)) is not None:
        raise HTTPException(
            status_code=409, detail=f"{repository!r} is already downloading as job {job_id}"
        )
    if repository in _DROPPING:
        raise HTTPException(status_code=409, detail=f"{repository!r} is being collected")
    # The folder rather than the catalog: an interrupted download leaves a `models--*` the
    # scan hides for being incomplete, and letting it through means paying for the whole
    # repository and only then failing on a rename into a directory that is already there.
    if (catalog.HUB_CACHE / _slug(repository)).exists():
        raise HTTPException(status_code=409, detail=f"{repository!r} is already on disk")
    job = registry.start(Download(model=repository), _download(repository))
    _DOWNLOADING[repository] = job.id
    return accepted(job)


@router.get("/admin/hub/staging")
def staged_repositories() -> list[Staged]:
    """The bytes a download left behind, and what they cost. One row per repository even when
    it has bytes in both places — which is what `_install` refusing to rename onto a name that
    appeared meanwhile leaves — because the repository is the unit the `DELETE` next to the row
    removes, and a second row for it would be a row whose button 404s."""
    totals: dict[str, int] = {}
    for path in _abandoned():
        repo = path.name.removeprefix("models--").replace("--", "/")
        totals[repo] = totals.get(repo, 0) + _bytes_on_disk(path)
    return [Staged(repo=repo, bytes_on_disk=total) for repo, total in totals.items()]


@router.delete("/admin/hub/staging/{repo:path}", status_code=204)
async def drop_staged(repo: str, registry: JobsDep) -> None:
    """Giving up on the resume. Refused while a job is fetching into it: `rmtree` under a
    running `hf_hub_download` is a download that dies on a directory that stopped existing,
    and `DELETE /admin/jobs/{id}` is how that one is stopped.

    On the loop up to the claim, and only then in a thread. A sync handler would run the whole
    thing in the threadpool, where the refusal above stops meaning anything: `create` runs on
    the loop and could register between the check and the `rmtree` walking the directory it
    just started fetching into.
    """
    if (job_id := _running(registry, repo)) is not None:
        raise HTTPException(status_code=409, detail=f"job {job_id} is downloading {repo!r}")
    slug = _slug(repo)
    directories = [path for path in _abandoned() if path.name == slug]
    if not directories:
        raise HTTPException(status_code=404, detail=f"nothing is staged for {repo!r}")
    _DROPPING.add(repo)
    try:
        for directory in directories:
            await asyncio.to_thread(shutil.rmtree, directory, True)
    finally:
        _DROPPING.discard(repo)


@router.get("/admin/hub/models")
def search(query: str, limit: int = 20) -> list[HubModel]:
    """The search field's source. Sorted by downloads: the field holds a guess at a name,
    and the popular match is nearly always the one meant."""
    return [
        HubModel(id=info.id, downloads=info.downloads, likes=info.likes)
        for info in HUB.list_models(search=query, sort="downloads", limit=limit)
    ]


def _info(repo: str) -> ModelInfo:
    try:
        return HUB.model_info(repo, files_metadata=True)
    except RepositoryNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"{repo!r} is not on the Hub") from error


@router.get("/admin/hub/models/{repo:path}/files")
def hub_files(repo: str) -> list[catalog.CheckpointFile]:
    """What `POST /admin/models` would fetch, priced by the Hub's own metadata: the
    pre-download twin of the catalog's files listing."""
    return sorted(
        (catalog.CheckpointFile(name=name, size=size) for name, size in _wanted(_info(repo))),
        key=lambda file: file.name,
    )


def _raw(repo: str) -> str | None:
    """The README as the Hub serves it. Private and absent answer alike to an anonymous
    reader (401), so anything but 200 is 'no card'."""
    answer = httpx.get(
        f"https://huggingface.co/{repo}/raw/main/README.md", follow_redirects=True, timeout=10
    )
    return answer.text if answer.status_code == 200 else None


@router.get("/admin/hub/models/{repo:path}/card", response_class=PlainTextResponse)
def hub_card(repo: str) -> str:
    """The repository's README raw, fetched over plain HTTP and never `hf_hub_download`:
    a cache entry for it would make `create` refuse the repository as already on disk."""
    text = _raw(repo)
    if text is None:
        raise HTTPException(status_code=404, detail=f"{repo!r} has no model card")
    return text
