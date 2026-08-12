"""The download job: what the catalog is allowed to see, and when.

The Hub is a double, and it writes the layout `hf_hub_download` writes — and, as of 1.24,
fails the way it fails: the partial file is unique to the attempt and is unlinked when the
attempt dies, so an interrupted file starts over and only whole files survive. A double that
resumed mid-file would make the resume test pass for a guarantee the dependency does not
give. Every gate the double stops at has a deadline: a download that never arrives fails the
test instead of hanging the suite.

The caches are `tmp_path`, always: nothing here may write into the machine's own hub cache,
and `catalog.HUB_CACHE` is a module constant precisely so a test can move it.
"""

import asyncio
import json
import shutil
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import huggingface_hub.errors
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from huggingface_hub import ModelInfo

from mlx_omnia.server import catalog, download, jobs, quantize
from mlx_omnia.server.store import Store

REPO = "mlx-community/Qwen3-0.6B-4bit"
TINY = "hf-internal-testing/tiny-random-gpt2"

_DEADLINE = 10.0
"""Every wait in this module is bounded by it — a job that never arrives fails here instead
of hanging the suite. Generous because a download of this shape is a few hundred rows in
sqlite, and the gate runs on a machine with three other suites behind it."""

CONFIG = json.dumps(
    {
        "model_type": "qwen3",
        "max_position_embeddings": 40960,
        "torch_dtype": "bfloat16",
        "quantization": {"group_size": 64, "bits": 4},
    }
).encode()

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = json.dumps({"weight_map": {"a": SHARDS[0], "b": SHARDS[1]}}).encode()

FILES: dict[str, bytes] = {
    "config.json": CONFIG,
    "model.safetensors.index.json": INDEX,
    SHARDS[0]: b"\0" * 4096,
    SHARDS[1]: b"\0" * 4096,
    "tokenizer.json": b"{}",
    # What the loader never opens, and what a download must therefore not pay for.
    "pytorch_model.bin": b"\1" * 8192,
    "original/consolidated.safetensors": b"\1" * 8192,
    "README.md": b"# a model card",
}

WANTED = ("config.json", "model.safetensors.index.json", *SHARDS, "tokenizer.json")
TOTAL = sum(len(FILES[name]) for name in WANTED)


@dataclass
class Hub:
    """The Hub as `download` sees it: a listing, a file-by-file fetch, and a search."""

    files: Mapping[str, bytes] = field(default_factory=lambda: FILES)
    sha: str = "0123456789abcdef"
    chunk: int = 64
    """Small enough that every file takes several, which is what lets a test stop the
    download inside one rather than between two."""
    served: int = 0
    """Bytes handed over, across every attempt: what a resume is measured by."""
    fetched: list[str] = field(default_factory=list)
    """The files this Hub actually served bytes for. A file answered off the cache does not
    appear — which is what per-file resume means."""
    asked: list[str] = field(default_factory=list)
    searched: list[tuple[str | None, str | None, int | None]] = field(default_factory=list)
    models: list[ModelInfo] = field(default_factory=list)
    reached: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=lambda: _open(threading.Event()))
    crash_after: int | None = None
    """Bytes after which the fetch dies the way a killed daemon does: mid-file."""
    swallow: bool = False
    """Answer a cancellation with a foreign exception, as a Rust frame would."""

    def pause(self) -> None:
        self.proceed.clear()

    def resume(self) -> None:
        self.proceed.set()

    def model_info(self, repo_id: str, *, files_metadata: bool, timeout: float = 0) -> ModelInfo:
        assert files_metadata, "sizes are what the progress total is summed from"
        self.asked.append(repo_id)
        return ModelInfo(
            id=repo_id,
            sha=self.sha,
            siblings=[
                {"rfilename": name, "size": len(content)} for name, content in self.files.items()
            ],
        )

    def list_models(
        self, *, search: str | None = None, sort: str | None = None, limit: int | None = None
    ) -> list[ModelInfo]:
        self.searched.append((search, sort, limit))
        return self.models

    def _arrive(self) -> None:
        """Where a test stops the download in the middle. The deadline is what turns a test
        that forgets to release into a failure instead of a suite that never ends."""
        self.reached.set()
        assert self.proceed.wait(_DEADLINE), "the download was never released"
        if self.crash_after is not None and self.served >= self.crash_after:
            raise ConnectionError("the daemon died")

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str,
        cache_dir: Path,
        tqdm_class: Callable[..., download._Bar],
    ) -> str:
        repository = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (repository / "blobs").mkdir(parents=True, exist_ok=True)
        target = snapshot / filename
        if target.is_file():
            # A file already in this cache: the hub answers with its path and no bytes.
            return str(target)
        content = self.files[filename]
        self.fetched.append(filename)
        # Unique to the attempt and unlinked when the attempt dies, exactly as
        # `_download_to_tmp_and_move` does it: nothing partial is ever picked back up.
        staged = repository / "blobs" / f"{filename}.{len(self.fetched)}.incomplete"
        written = 0
        try:
            with tqdm_class(total=len(content), initial=0) as bar, staged.open("wb") as handle:
                while written < len(content):
                    piece = content[written : written + self.chunk]
                    handle.write(piece)
                    handle.flush()
                    written += len(piece)
                    self.served += len(piece)
                    try:
                        bar.update(len(piece))
                    except jobs.Cancelled:
                        if not self.swallow:
                            raise
                        raise RuntimeError("xet: reconstruction failed") from None
                    self._arrive()
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        staged.rename(repository / "blobs" / filename)
        target.symlink_to(Path("..") / ".." / "blobs" / filename)
        return str(target)


def _open(event: threading.Event) -> threading.Event:
    event.set()
    return event


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(catalog, "HUB_CACHE", tmp_path / "hub")
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    return tmp_path / "hub"


@pytest.fixture
def hub(caches: Path, monkeypatch: pytest.MonkeyPatch) -> Hub:
    double = Hub()
    monkeypatch.setattr(download, "HUB", double)
    monkeypatch.setattr(download, "hf_hub_download", double.download)
    # The interval exists so a 30 GB download is not a sqlite transaction per chunk; here it
    # would only mean the frame a test reads is the one from before the chunk it waited for.
    monkeypatch.setattr(download, "_REPORT_SECONDS", 0.0)
    return double


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(download.router)
    app.include_router(jobs.router)
    app.state.jobs = jobs.Jobs(Store(tmp_path / "server.db"))
    with TestClient(app) as running:
        yield running


def start(client: TestClient, **body: str) -> str:
    response = client.post("/admin/models", json=body)
    assert response.status_code == 202, response.text
    assert response.headers["location"].endswith(response.json()["id"])
    # The jobs screen filters by kind, so the string is a contract and not a label.
    assert response.json()["kind"] == "download"
    job_id = response.json()["id"]
    assert isinstance(job_id, str)
    return job_id


def view(client: TestClient, job_id: str) -> dict[str, object]:
    response = client.get(f"/admin/jobs/{job_id}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def wait_for(
    client: TestClient, job_id: str, state: str, seconds: float = _DEADLINE
) -> dict[str, object]:
    deadline = time.monotonic() + seconds
    while True:
        current = view(client, job_id)
        if current["state"] == state:
            return current
        assert time.monotonic() < deadline, (
            f"job stayed in {current['state']!r} ({current['error']!r}), wanted {state!r}"
        )
        time.sleep(0.01)


def progress(current: Mapping[str, object]) -> tuple[float, float]:
    frame = current["progress"]
    assert isinstance(frame, dict)
    completed, total = frame["completed"], frame["total"]
    assert isinstance(completed, int | float) and isinstance(total, int | float)
    return completed, total


def repositories(root: Path) -> list[Path]:
    """What the catalog globs: only the final names, never the staging directory."""
    return list(root.glob("models--*"))


def test_a_download_reports_its_bytes_and_only_then_enters_the_catalog(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """The two halves of the contract in one run: the screen has real bytes to draw while
    the download is going, and the catalog has nothing to offer until it is over."""
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the download never began"

    running = view(client, job_id)
    completed, total = progress(running)
    assert running["state"] == "running"
    assert total == TOTAL, "the total is summed over the files the loader reads"
    assert 0 < completed < total
    assert catalog.scan() == []
    assert repositories(caches) == []

    hub.resume()
    finished = wait_for(client, job_id, "ok")

    assert progress(finished) == (TOTAL, TOTAL)
    assert hub.served == TOTAL, "a byte was paid for twice"
    entries = catalog.scan()
    assert [entry.id for entry in entries] == [REPO]
    assert sorted(path.name for path in entries[0].directory.iterdir()) == sorted(WANTED)
    assert repositories(caches / download._STAGING) == []


def test_a_download_that_dies_midway_leaves_no_entry_and_the_next_one_resumes(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """A daemon killed inside a shard is the ordinary case, not the exotic one. What it must
    not leave behind is a snapshot under its final name; what it must not pay for twice are
    the files that had already landed.

    The shard it died inside does start over — `hf_hub_download` keeps no reusable partial —
    so the guarantee is per file and not per byte, and this is where that is written down.
    """
    hub.crash_after = len(CONFIG) + len(INDEX) + 2048
    died = wait_for(client, start(client, repo=REPO), "error")

    assert died["error"] == "ConnectionError: the daemon died"
    landed = {"config.json", "model.safetensors.index.json"}
    assert landed <= set(hub.fetched), "the crash came before anything could land"
    assert catalog.scan() == []
    assert repositories(caches) == []
    assert repositories(caches / download._STAGING) != [], "nothing was left to resume from"

    hub.crash_after = None
    hub.fetched.clear()
    already = hub.served
    finished = wait_for(client, start(client, repo=REPO), "ok")

    assert landed.isdisjoint(hub.fetched), "a file that had already landed was fetched again"
    assert hub.served - already < TOTAL, "the second attempt paid for the whole repository"
    assert progress(finished) == (TOTAL, TOTAL), "a file taken off the cache still counts"
    assert [entry.id for entry in catalog.scan()] == [REPO]


def test_a_delete_stops_the_download_and_no_snapshot_gets_its_final_name(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """`asyncio.Task.cancel()` reaches neither a socket read nor the thread the download runs
    in. What stops it is the flag read on the way past every chunk — and a download stopped
    halfway must leave the catalog exactly as empty as it found it."""
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the download never began"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None
    assert hub.served < TOTAL, "the download ran to the end and only then noticed"
    assert catalog.scan() == []
    assert repositories(caches) == []
    assert repositories(caches / download._STAGING) == [], "the stopped download kept its bytes"


def test_a_cancellation_that_comes_back_as_another_exception_still_ends_cancelled(
    client: TestClient, hub: Hub
) -> None:
    """The xet path runs the byte callback inside a Rust frame, and what crosses back may be
    any exception carrying the message. A `DELETE` answered with `error` is a `DELETE` the
    screen reports as a failure."""
    hub.swallow = True
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the download never began"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None


def test_a_snapshot_missing_a_shard_of_its_weight_map_never_gets_its_final_name(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """The failure 32.2 exists to prevent: an index naming a shard nobody delivered. The
    download succeeds file by file and the rename is what refuses, so the catalog never
    offers a model that fails at load."""
    hub.files = {name: content for name, content in FILES.items() if name != SHARDS[1]}
    failed = wait_for(client, start(client, repo=REPO), "error")

    error = failed["error"]
    assert isinstance(error, str) and SHARDS[1] in error
    assert catalog.scan() == []
    assert repositories(caches) == []
    staged = caches / download._STAGING / f"models--{REPO.replace('/', '--')}"
    assert (staged / "snapshots" / hub.sha / SHARDS[0]).is_file()


def test_a_second_download_of_the_same_repository_is_refused_while_the_first_runs(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """Staging is per repository, so two jobs on one repository are two jobs writing into one
    directory: whichever finishes first renames it away, and the other reports a missing shard
    for a download that in fact arrived. The second `POST` goes out while the first job is
    demonstrably inside a file — `reached` is set from between two chunks — because a guard
    that is only tested before the download begins is a guard tested against nothing."""
    hub.pause()
    first = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the download never began"

    response = client.post("/admin/models", json={"repo": REPO})

    assert response.status_code == 409, response.text
    assert first in response.json()["detail"], "the refusal must name the job to watch"
    assert len(client.get("/admin/jobs").json()) == 1, "a second job was started anyway"

    hub.resume()
    finished = wait_for(client, first, "ok")

    assert progress(finished) == (TOTAL, TOTAL)
    assert hub.served == TOTAL, "one repository was fetched twice into one staging directory"
    assert [entry.id for entry in catalog.scan()] == [REPO]
    assert repositories(caches / download._STAGING) == []


def test_a_cancelled_download_collects_the_staging_and_the_next_one_starts_from_nothing(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """A failure keeps its bytes so the next attempt can resume; a cancellation is the ending
    that says the repository is not wanted, and it takes the staging with it — including what
    an earlier failed attempt had banked, since the staging belongs to the repository and not
    to the job. Without that, a `DELETE` on a 60 GB download leaves 60 GB nobody ever looks
    at again."""
    hub.crash_after = len(CONFIG) + len(INDEX) + 2048
    wait_for(client, start(client, repo=REPO), "error")
    banked = {"config.json", "model.safetensors.index.json"}
    assert banked <= set(hub.fetched), "the crash came before anything could land"
    assert repositories(caches / download._STAGING) != [], "there was nothing to collect"

    hub.crash_after = None
    hub.reached.clear()
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the second download never began"
    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()

    assert wait_for(client, job_id, "cancelled")["error"] is None
    assert repositories(caches / download._STAGING) == []

    hub.fetched.clear()
    wait_for(client, start(client, repo=REPO), "ok")

    assert set(hub.fetched) == set(WANTED), "the cancelled job's staging outlived it"
    assert [entry.id for entry in catalog.scan()] == [REPO]


def test_a_cancel_taken_after_the_last_file_still_takes_the_staging_with_it(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """The collection is a claim about the ending, so an exit it does not cover is bytes
    abandoned — and the exit past the last file is the worst of them, the one moment the
    staging holds the whole repository. Cancelled while the only file is in flight, the job
    leaves through the report that follows the loop rather than through the loop itself; with
    the collection scoped to the loop alone, this job ends `cancelled` with its bytes intact.
    """
    hub.files = {"config.json": CONFIG}
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the download never began"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()

    assert wait_for(client, job_id, "cancelled")["error"] is None
    assert hub.fetched == ["config.json"], "the cancel landed somewhere other than the last file"
    assert repositories(caches / download._STAGING) == [], "the staging outlived the cancellation"


def test_the_staging_a_failed_download_left_is_listed_with_its_bytes_and_can_be_dropped(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """The other half of the policy: what a failure keeps is kept for the resume, and it is
    kept in the open. Two files landed whole and the shard the crash interrupted left nothing,
    so the number is exactly those two — counted once, because the snapshot next to them is
    symlinks into the same blobs."""
    hub.crash_after = len(CONFIG) + len(INDEX) + 2048
    wait_for(client, start(client, repo=REPO), "error")

    listed = client.get("/admin/hub/staging")

    assert listed.status_code == 200, listed.text
    assert listed.json() == [{"repo": REPO, "bytes_on_disk": len(CONFIG) + len(INDEX)}]

    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 204

    assert client.get("/admin/hub/staging").json() == []
    assert repositories(caches / download._STAGING) == []
    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 404


def test_the_staging_of_a_running_download_is_not_dropped_from_under_it(
    client: TestClient, hub: Hub
) -> None:
    """Dropping the staging is for bytes nobody is fetching. Unlinking the directory a running
    `hf_hub_download` is writing into fails the download somewhere it cannot explain, and the
    call that stops a download is `DELETE /admin/jobs/{id}`."""
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(_DEADLINE), "the download never began"

    response = client.delete(f"/admin/hub/staging/{REPO}")

    assert response.status_code == 409, response.text
    assert job_id in response.json()["detail"]

    hub.resume()
    finished = wait_for(client, job_id, "ok")

    assert progress(finished) == (TOTAL, TOTAL), "the running download lost its staging"


def test_a_download_is_refused_while_the_staging_it_would_write_into_is_collected(
    hub: Hub, caches: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of that refusal is a claim, and this is the window it covers: `create`
    runs on the loop and the `rmtree` does not, so from the moment the collection goes to a
    thread until the last file is gone there is an event loop free to accept a download into
    the directory being walked away. What that leaves is a snapshot whose symlinks point at
    blobs the collection removed — whole to the rename gate, empty to the loader.

    Driven without the client: what has to happen at the same time is a handler on the loop
    and a handler in the threadpool, and the `rmtree` is where the second one is.
    """
    staged = caches / download._STAGING / f"models--{REPO.replace('/', '--')}"
    snapshot = staged / "snapshots" / hub.sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(CONFIG)
    real = shutil.rmtree
    answers: list[int] = []

    async def scenario() -> None:
        registry = jobs.Jobs(Store(tmp_path / "collecting.db"))
        loop = asyncio.get_running_loop()

        async def start_a_download() -> int:
            try:
                await download.create(download.DownloadRequest(repo=REPO), registry)
            except HTTPException as refused:
                return refused.status_code
            return 202

        def collecting(directory: Path, ignore_errors: bool = False) -> None:
            answers.append(
                asyncio.run_coroutine_threadsafe(start_a_download(), loop).result(_DEADLINE)
            )
            real(directory, ignore_errors=ignore_errors)

        monkeypatch.setattr(shutil, "rmtree", collecting)
        await download.drop_staged(REPO, registry)

    asyncio.run(scenario())

    assert answers == [409], "a download began fetching into a directory being collected"
    assert not staged.exists(), "the collection did not finish"


def test_a_repository_staged_in_both_places_is_one_row_and_one_delete(
    client: TestClient, caches: Path
) -> None:
    """Bytes in the staging cache and an incomplete `models--*` in the cache proper is not a
    contrived pair: it is what `_install` leaves when the final name appeared while the
    download was running, and it names one repository the user gave up on. Two rows for it is
    two buttons where the first makes the second 404, since the `DELETE` already takes the
    repository — not the directory — as what it collects."""
    slug = f"models--{REPO.replace('/', '--')}"
    staged = caches / download._STAGING / slug / "snapshots" / "abc"
    staged.mkdir(parents=True)
    (staged / "config.json").write_bytes(CONFIG)
    orphan = caches / slug / "snapshots" / "abc"
    orphan.mkdir(parents=True)
    (orphan / "model.safetensors.index.json").write_bytes(INDEX)

    listed = client.get("/admin/hub/staging")

    assert listed.status_code == 200, listed.text
    assert listed.json() == [{"repo": REPO, "bytes_on_disk": len(CONFIG) + len(INDEX)}]

    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 204

    assert not (caches / download._STAGING / slug).exists(), "the staging cache survived"
    assert not (caches / slug).exists(), "the incomplete repository survived"


@pytest.mark.parametrize(
    ("weights", "complete"),
    [
        pytest.param(
            {"model.safetensors.index.json": INDEX, SHARDS[0]: b"", SHARDS[1]: b""},
            True,
            id="every-shard-of-the-index",
        ),
        pytest.param(
            {"model.safetensors.index.json": INDEX, SHARDS[0]: b""},
            False,
            id="one-shard-short",
        ),
        pytest.param(
            {"model.safetensors.index.json": INDEX, SHARDS[0]: b"", SHARDS[1]: None},
            False,
            id="a-pointer-to-a-blob-that-never-arrived",
        ),
        pytest.param({"model.safetensors": b""}, True, id="a-single-unsharded-file"),
        pytest.param({}, False, id="no-weights-at-all"),
    ],
)
def test_the_rename_gate_and_the_catalog_agree_on_which_snapshots_are_complete(
    caches: Path, weights: Mapping[str, bytes | None], complete: bool
) -> None:
    """`_install` decides whether a snapshot may take its final name; `catalog._complete`
    decides whether that same snapshot is a model worth listing. They are two readings of one
    rule — the `weight_map` — and no line of code holds them together. What this test loses is
    the day one of them moves: the download starts publishing what the catalog hides, or
    refusing what it would list, and neither says so.

    The weights are the axis they must agree on, and the only one: every case here writes
    `config.json`, which is the axis where they part on purpose (the test below)."""
    snapshot = caches / f"models--{REPO.replace('/', '--')}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(CONFIG)
    for name, content in weights.items():
        if content is None:
            (snapshot / name).symlink_to(snapshot / "blobs" / name)
        else:
            (snapshot / name).write_bytes(content)

    assert (download._missing(snapshot) == []) is complete
    assert catalog._complete(snapshot) is complete
    assert [entry.id for entry in catalog.scan()] == ([REPO] if complete else [])


def test_a_snapshot_with_every_weight_and_no_config_is_refused_and_would_be_hidden(
    caches: Path,
) -> None:
    """The one axis on which the two readings part, and part on purpose: `_missing` also
    demands `config.json`, `catalog._complete` reads the `weight_map` alone, and what hides
    such a snapshot from the catalog is `_entry` further down. Without this the `config.json`
    line in `_missing` can be deleted with the suite still green, and the failure it exists to
    prevent is a job that finishes `ok` over a repository the catalog then does not list."""
    snapshot = caches / f"models--{REPO.replace('/', '--')}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"")

    assert download._missing(snapshot) == ["config.json"], "the rename gate let it through"
    assert catalog._complete(snapshot) is True, "the weights are all there"
    assert catalog.scan() == [], "a snapshot with no config.json was listed"


def test_a_repository_already_on_disk_is_refused_instead_of_downloaded_again(
    client: TestClient, hub: Hub
) -> None:
    wait_for(client, start(client, repo=REPO), "ok")

    response = client.post("/admin/models", json={"repo": REPO})

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    # Both refusals name the repository, so naming it does not say which one arrived: this
    # test is about the one on disk, and the download-already-running 409 would pass too.
    assert REPO in detail and "already on disk" in detail, detail
    assert len(client.get("/admin/jobs").json()) == 1, "a second job was started anyway"
    assert hub.served == TOTAL


def test_a_repository_left_incomplete_in_the_hub_cache_is_refused_too(
    client: TestClient, caches: Path
) -> None:
    """`load` downloads into the same hub cache (`mlx_omnia/task.py`), so a `load` killed
    partway leaves a `models--*` that `catalog.scan` refuses to list for being incomplete.
    A check that reads the catalog lets it through, pays for the whole repository, and only
    then fails renaming staging onto a directory that is already there — every time."""
    (caches / f"models--{REPO.replace('/', '--')}" / "snapshots" / "abc").mkdir(parents=True)

    response = client.post("/admin/models", json={"repo": REPO})

    assert response.status_code == 409, response.text
    assert client.get("/admin/jobs").json() == [], "a job was started for a doomed download"


def test_the_incomplete_repository_the_refusal_names_can_be_collected_and_fetched_again(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """The other side of that refusal, and what makes it a refusal rather than a dead end. The
    directory is refused by `create`, hidden from `catalog.scan` and therefore unreachable by
    `catalog.remove`: with no door of its own the repository can never be downloaded again
    through the API, only by deleting the folder by hand."""
    orphan = caches / f"models--{REPO.replace('/', '--')}"
    (orphan / "snapshots" / "abc").mkdir(parents=True)
    (orphan / "snapshots" / "abc" / "config.json").write_bytes(CONFIG)

    listed = client.get("/admin/hub/staging")

    assert listed.status_code == 200, listed.text
    assert [item["repo"] for item in listed.json()] == [REPO]
    assert listed.json()[0]["bytes_on_disk"] == len(CONFIG)

    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 204
    assert not orphan.exists(), "the incomplete repository survived the collection"

    wait_for(client, start(client, repo=REPO), "ok")
    assert [entry.id for entry in catalog.scan()] == [REPO]


def test_quant_asks_the_hub_for_the_variant_someone_already_published(
    client: TestClient, hub: Hub
) -> None:
    """The decision this task closed: `quant` selects a repository that exists on the Hub,
    verified by whoever uploaded it and a fraction of the bytes. Quantizing at home is 34.3,
    and no path here does it."""
    wait_for(client, start(client, repo="Qwen/Qwen3-0.6B", quant="4bit"), "ok")

    assert hub.asked == ["mlx-community/Qwen3-0.6B-4bit"]
    assert [entry.id for entry in catalog.scan()] == ["mlx-community/Qwen3-0.6B-4bit"]


def test_the_search_field_asks_the_hub_for_what_was_typed(client: TestClient, hub: Hub) -> None:
    """Tested against the double rather than the Hub: what this route owns is the query it
    sends and the three fields the screen draws, and neither depends on the network."""
    hub.models = [
        ModelInfo(id="mlx-community/Qwen3-0.6B-4bit", downloads=1200, likes=7),
        ModelInfo(id="Qwen/Qwen3-0.6B", downloads=900, likes=None),
    ]

    response = client.get("/admin/hub/models", params={"query": "qwen3", "limit": 2})

    assert response.status_code == 200, response.text
    assert hub.searched == [("qwen3", "downloads", 2)]
    assert response.json() == [
        {"id": "mlx-community/Qwen3-0.6B-4bit", "downloads": 1200, "likes": 7},
        {"id": "Qwen/Qwen3-0.6B", "downloads": 900, "likes": None},
    ]


def test_the_hub_files_listing_is_what_the_download_would_fetch(
    client: TestClient, hub: Hub
) -> None:
    """The pre-download listing prices exactly `_wanted` — the `.bin` and the subfolder the
    job never pays for are not offered as if they cost something."""
    response = client.get(f"/admin/hub/models/{REPO}/files")

    assert response.status_code == 200, response.text
    assert response.json() == sorted(
        ({"name": name, "size": len(FILES[name])} for name in WANTED),
        key=lambda entry: entry["name"],
    )


def test_a_repository_the_hub_does_not_know_is_a_404(
    client: TestClient, hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(repo_id: str, *, files_metadata: bool, timeout: float = 0) -> ModelInfo:
        raise huggingface_hub.errors.RepositoryNotFoundError(
            "gone", response=httpx.Response(404, request=httpx.Request("GET", "https://hf.co"))
        )

    monkeypatch.setattr(hub, "model_info", gone)

    response = client.get("/admin/hub/models/nobody/nothing/files")

    assert response.status_code == 404


def test_the_hub_card_is_the_raw_readme_and_absent_is_a_404(
    client: TestClient, hub: Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raw(repo: str) -> str | None:
        return "# a card" if repo == REPO else None

    monkeypatch.setattr(download, "_raw", raw)

    served = client.get(f"/admin/hub/models/{REPO}/card")
    missing = client.get("/admin/hub/models/other/repo/card")

    assert (served.status_code, served.text) == (200, "# a card")
    assert missing.status_code == 404


def test_a_real_repository_arrives_whole_and_the_catalog_reads_it(
    client: TestClient, caches: Path
) -> None:
    """The one test that talks to the Hub, over a 0.5 MB repository. No double can show that
    the layout `hf_hub_download` writes is the layout `catalog.scan` reads — the symlinks
    into `blobs/` survived the rename, the `.bin` and the `.h5` were never fetched — and that
    is the whole of what this covers. Skipped when the Hub is unreachable: offline is not a
    regression.
    """
    try:
        # Only unreachable: a signature that moved, or a `TypeError` out of the tqdm
        # contract this module leans on, has to fail here rather than pass as a skip.
        download.HUB.model_info(TINY, files_metadata=True, timeout=10)
    except (OSError, huggingface_hub.errors.HfHubHTTPError) as error:
        pytest.skip(f"the Hub is not reachable: {error}")

    finished = wait_for(client, start(client, repo=TINY), "ok", seconds=120)

    entries = catalog.scan()
    assert [entry.id for entry in entries] == [TINY]
    completed, total = progress(finished)
    on_disk = sum(path.stat().st_size for path in entries[0].directory.iterdir() if path.is_file())
    # The headers said one thing and the disk another only if the fetch dropped a file.
    assert completed == total == on_disk
    assert entries[0].architecture == "gpt2"
    assert (entries[0].directory / "model.safetensors").is_file()
    assert not (entries[0].directory / "pytorch_model.bin").exists()
    assert not (entries[0].directory / "tf_model.h5").exists()
    assert repositories(caches / download._STAGING) == []


def test_the_staging_a_killed_quantization_left_is_listed_and_dropped_here_too(
    client: TestClient, caches: Path
) -> None:
    """A quantization that dies hard leaves tens of gigabytes under `.quantizing`, which
    neither the catalog nor `hf cache scan` shows — a dot-directory nothing lists and nothing
    deletes. What a caller is asking here is what is taking space and can be dropped, and
    which job wrote the bytes is not part of that question."""
    left = caches / quantize.STAGING / f"models--{REPO.replace('/', '--')}"
    left.mkdir(parents=True)
    (left / "model.safetensors").write_bytes(b"x" * 4096)

    listed = client.get("/admin/hub/staging")

    assert listed.status_code == 200, listed.text
    assert listed.json() == [{"repo": REPO, "bytes_on_disk": 4096}]

    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 204
    assert not left.exists()
    assert client.get("/admin/hub/staging").json() == []
