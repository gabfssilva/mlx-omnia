"""What a download leaves on disk when it does not finish: who lists it, who collects it,
and which requests it refuses in the meantime."""

import asyncio
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_omnia.server.services import catalog, downloads, quantize
from mlx_omnia.server.services.downloads.staging import STAGING
from mlx_omnia.server.services.jobs import Jobs
from tests.server.download_stand import (
    CONFIG,
    DEADLINE,
    INDEX,
    REPO,
    TOTAL,
    Hub,
    caches,
    client,
    hub,
    repositories,
    start,
)
from tests.server.polling import progress, wait_for

__all__ = ["caches", "client", "hub"]
"""The fixtures this module runs on, imported rather than repeated."""


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
    assert repositories(caches / STAGING) == []
    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 404


def test_the_staging_of_a_running_download_is_not_dropped_from_under_it(
    client: TestClient, hub: Hub
) -> None:
    """Dropping the staging is for bytes nobody is fetching. Unlinking the directory a running
    `hf_hub_download` is writing into fails the download somewhere it cannot explain, and the
    call that stops a download is `DELETE /admin/jobs/{id}`."""
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(DEADLINE), "the download never began"

    response = client.delete(f"/admin/hub/staging/{REPO}")

    assert response.status_code == 409, response.text
    assert job_id in response.json()["detail"]

    hub.resume()
    finished = wait_for(client, job_id, "ok")

    assert progress(finished) == (TOTAL, TOTAL), "the running download lost its staging"


def test_a_download_is_refused_while_the_staging_it_would_write_into_is_collected(
    hub: Hub, caches: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of that refusal is a claim, and this is the window it covers:
    `start_download` runs on the loop and the `rmtree` does not, so from the moment the
    collection goes to a thread until the last file is gone there is an event loop free to
    accept a download into the directory being walked away. What that leaves is a snapshot
    whose symlinks point at blobs the collection removed — whole to the rename gate, empty to
    the loader.

    Driven without the client: what has to happen at the same time is a handler on the loop
    and a handler in the threadpool, and the `rmtree` is where the second one is.
    """
    staged = caches / STAGING / f"models--{REPO.replace('/', '--')}"
    snapshot = staged / "snapshots" / hub.sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(CONFIG)
    real = shutil.rmtree
    answers: list[int] = []

    async def scenario() -> None:
        registry = Jobs()
        loop = asyncio.get_running_loop()

        async def start_a_download() -> int:
            try:
                await downloads.start_download(registry, REPO)
            except downloads.BeingCollected:
                return 409
            except downloads.AlreadyDownloading:
                return 409
            except downloads.AlreadyOnDisk:
                return 409
            return 202

        def collecting(directory: Path, ignore_errors: bool = False) -> None:
            answers.append(
                asyncio.run_coroutine_threadsafe(start_a_download(), loop).result(DEADLINE)
            )
            real(directory, ignore_errors=ignore_errors)

        monkeypatch.setattr(shutil, "rmtree", collecting)
        await downloads.drop_staged(registry, REPO)

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
    staged = caches / STAGING / slug / "snapshots" / "abc"
    staged.mkdir(parents=True)
    (staged / "config.json").write_bytes(CONFIG)
    orphan = caches / slug / "snapshots" / "abc"
    orphan.mkdir(parents=True)
    (orphan / "model.safetensors.index.json").write_bytes(INDEX)

    listed = client.get("/admin/hub/staging")

    assert listed.status_code == 200, listed.text
    assert listed.json() == [{"repo": REPO, "bytes_on_disk": len(CONFIG) + len(INDEX)}]

    assert client.delete(f"/admin/hub/staging/{REPO}").status_code == 204

    assert not (caches / STAGING / slug).exists(), "the staging cache survived"
    assert not (caches / slug).exists(), "the incomplete repository survived"


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
    directory is refused by `start_download`, hidden from `catalog.scan` and therefore
    unreachable by `catalog.remove`: with no door of its own the repository can never be
    downloaded again through the API, only by deleting the folder by hand."""
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
