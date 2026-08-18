"""The rules the download and the catalog share about a finished snapshot, and the read-only
window on the Hub the download screen draws itself from."""

from collections.abc import Mapping
from pathlib import Path

import httpx
import huggingface_hub.errors
import pytest
from fastapi.testclient import TestClient
from huggingface_hub import ModelInfo

from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.downloads import hub as hub_service
from mlx_omnia.server.services.downloads.hub import missing
from mlx_omnia.server.services.downloads.staging import STAGING
from tests.server.download_stand import (
    CONFIG,
    FILES,
    INDEX,
    REPO,
    SHARDS,
    TINY,
    WANTED,
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
    """`_install` decides whether a snapshot may take its final name; the catalog's own
    completeness decides whether that same snapshot is a model worth listing. They are two
    readings of one rule — the `weight_map` — and no line of code holds them together. What
    this test loses is the day one of them moves: the download starts publishing what the
    catalog hides, or refusing what it would list, and neither says so.

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

    assert (missing(snapshot) == []) is complete
    assert catalog._complete(snapshot) is complete
    assert [entry.id for entry in catalog.scan()] == ([REPO] if complete else [])


def test_a_snapshot_with_every_weight_and_no_config_is_refused_and_would_be_hidden(
    caches: Path,
) -> None:
    """The one axis on which the two readings part, and part on purpose: `missing` also
    demands `config.json`, the catalog's completeness reads the `weight_map` alone, and what
    hides such a snapshot from the catalog is its entry reader further down. Without this the
    `config.json` line in `missing` can be deleted with the suite still green, and the failure
    it exists to prevent is a job that finishes `ok` over a repository the catalog then does
    not list."""
    snapshot = caches / f"models--{REPO.replace('/', '--')}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"")

    assert missing(snapshot) == ["config.json"], "the rename gate let it through"
    assert catalog._complete(snapshot) is True, "the weights are all there"
    assert catalog.scan() == [], "a snapshot with no config.json was listed"


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
    """The pre-download listing prices exactly what is wanted — the `.bin` and the subfolder
    the job never pays for are not offered as if they cost something."""
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

    monkeypatch.setattr(hub_service, "_raw", raw)

    served = client.get(f"/admin/hub/models/{REPO}/card")
    missing_card = client.get("/admin/hub/models/other/repo/card")

    assert (served.status_code, served.text) == (200, "# a card")
    assert missing_card.status_code == 404


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
        #
        # `httpx.TransportError` is here because huggingface_hub's session is httpx now: a
        # DNS failure arrives as `httpx.ConnectError`, which is not an `OSError`, so the
        # offline skip this docstring promises did not hold. Its sibling `HTTPStatusError`
        # stays out — a status the Hub answered is not the Hub being unreachable.
        hub_service.HUB.model_info(TINY, files_metadata=True, timeout=10)
    except (OSError, httpx.TransportError, huggingface_hub.errors.HfHubHTTPError) as error:
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
    assert repositories(caches / STAGING) == []
