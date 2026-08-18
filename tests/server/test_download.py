"""The download job: what the catalog is allowed to see, and when.

The stand — the Hub double, the caches and the client — is `download_stand.py`; what is here
is the job itself, from the first byte to the ending that decides what happens to the bytes.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.downloads.staging import STAGING
from tests.server.download_stand import (
    CONFIG,
    DEADLINE,
    FILES,
    INDEX,
    REPO,
    SHARDS,
    TOTAL,
    WANTED,
    Hub,
    caches,
    client,
    hub,
    repositories,
    start,
)
from tests.server.polling import progress, view, wait_for

__all__ = ["caches", "client", "hub"]
"""The fixtures this module runs on, imported rather than repeated."""


def test_a_download_reports_its_bytes_and_only_then_enters_the_catalog(
    client: TestClient, hub: Hub, caches: Path
) -> None:
    """The two halves of the contract in one run: the screen has real bytes to draw while
    the download is going, and the catalog has nothing to offer until it is over."""
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(DEADLINE), "the download never began"

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
    assert repositories(caches / STAGING) == []


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
    assert repositories(caches / STAGING) != [], "nothing was left to resume from"

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
    assert hub.reached.wait(DEADLINE), "the download never began"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()
    cancelled = wait_for(client, job_id, "cancelled")

    assert cancelled["error"] is None
    assert hub.served < TOTAL, "the download ran to the end and only then noticed"
    assert catalog.scan() == []
    assert repositories(caches) == []
    assert repositories(caches / STAGING) == [], "the stopped download kept its bytes"


def test_a_cancellation_that_comes_back_as_another_exception_still_ends_cancelled(
    client: TestClient, hub: Hub
) -> None:
    """The xet path runs the byte callback inside a Rust frame, and what crosses back may be
    any exception carrying the message. A `DELETE` answered with `error` is a `DELETE` the
    screen reports as a failure."""
    hub.swallow = True
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(DEADLINE), "the download never began"

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
    staged = caches / STAGING / f"models--{REPO.replace('/', '--')}"
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
    assert hub.reached.wait(DEADLINE), "the download never began"

    response = client.post("/admin/models", json={"repo": REPO})

    assert response.status_code == 409, response.text
    assert first in response.json()["detail"], "the refusal must name the job to watch"
    assert len(client.get("/admin/jobs").json()) == 1, "a second job was started anyway"

    hub.resume()
    finished = wait_for(client, first, "ok")

    assert progress(finished) == (TOTAL, TOTAL)
    assert hub.served == TOTAL, "one repository was fetched twice into one staging directory"
    assert [entry.id for entry in catalog.scan()] == [REPO]
    assert repositories(caches / STAGING) == []


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
    assert repositories(caches / STAGING) != [], "there was nothing to collect"

    hub.crash_after = None
    hub.reached.clear()
    hub.pause()
    job_id = start(client, repo=REPO)
    assert hub.reached.wait(DEADLINE), "the second download never began"
    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()

    assert wait_for(client, job_id, "cancelled")["error"] is None
    assert repositories(caches / STAGING) == []

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
    assert hub.reached.wait(DEADLINE), "the download never began"

    assert client.delete(f"/admin/jobs/{job_id}").status_code == 202
    hub.resume()

    assert wait_for(client, job_id, "cancelled")["error"] is None
    assert hub.fetched == ["config.json"], "the cancel landed somewhere other than the last file"
    assert repositories(caches / STAGING) == [], "the staging outlived the cancellation"
