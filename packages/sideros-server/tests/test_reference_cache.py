"""The reference cache's index and its lifecycle. The pass that fills it is task 59.9's
`# TODO`; what is under test is everything around it, because that is the part the routes
have to be honest about — a cache nobody can see is rubbish accumulating in silence."""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sideros_server import fidelity
from sideros_server.store import Store


@pytest.fixture
def stand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[TestClient, Store]]:
    monkeypatch.setattr(fidelity, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    store = Store(tmp_path / "server.db")
    app = FastAPI()
    app.state.store = store
    app.include_router(fidelity.router)
    with TestClient(app) as client:
        yield client, store


def _entry(store: Store, reference: str = "house/bf16", corpus: str = "wikitext103") -> str:
    path = fidelity.cache_path(reference, corpus, 10000, 42)
    path.write_bytes(b"\0" * 5_120_000)
    return fidelity.record(store, reference, corpus, 10000, 42, path).id


def test_the_listing_says_what_the_cache_costs_on_disk(
    stand: tuple[TestClient, Store],
) -> None:
    client, store = stand
    _entry(store, "house/bf16")
    _entry(store, "house/other")

    body = client.get("/admin/benchmarks/references").json()

    assert len(body["entries"]) == 2
    assert body["total_bytes"] == 2 * 5_120_000
    assert all(entry["present"] for entry in body["entries"])
    assert body["entries"][0]["topk"] == 64


def test_deleting_takes_the_row_and_the_file(stand: tuple[TestClient, Store]) -> None:
    client, store = stand
    entry_id = _entry(store)
    path = fidelity.cache_path("house/bf16", "wikitext103", 10000, 42)
    assert path.is_file()

    assert client.delete(f"/admin/benchmarks/references/{entry_id}").status_code == 204

    assert not path.exists()
    assert store.reference(entry_id) is None
    assert client.delete(f"/admin/benchmarks/references/{entry_id}").status_code == 404


def test_a_row_whose_file_is_gone_is_not_a_cache_hit(stand: tuple[TestClient, Store]) -> None:
    """The disk is the truth. The row is dropped so the next pass rebuilds it instead of
    reading a path that is not there."""
    client, store = stand
    _entry(store)
    fidelity.cache_path("house/bf16", "wikitext103", 10000, 42).unlink()

    assert client.get("/admin/benchmarks/references").json()["total_bytes"] == 0
    assert fidelity.existing(store, "house/bf16", "wikitext103", 10000, 42) is None
    assert store.references() == []


def test_an_entry_already_on_disk_is_reused_instead_of_rebuilt(
    stand: tuple[TestClient, Store],
) -> None:
    """Which is the whole point: one expensive pass, then five cheap comparisons."""
    _client, store = stand
    entry_id = _entry(store)

    assert fidelity.build(store, "house/bf16", "wikitext103", 10000, 42).id == entry_id


def test_building_a_reference_that_is_not_cached_says_which_task_fills_it(
    stand: tuple[TestClient, Store],
) -> None:
    _client, store = stand

    with pytest.raises(NotImplementedError, match=r"59\.9"):
        fidelity.build(store, "house/bf16", "wikitext103", 10000, 42)


def test_the_key_is_the_four_things_that_change_the_answer(
    stand: tuple[TestClient, Store],
) -> None:
    """Change any of them and the cached logits answer a different question."""
    base = fidelity.cache_key("house/bf16", "wikitext103", 10000, 42)

    assert fidelity.cache_key("house/other", "wikitext103", 10000, 42) != base
    assert fidelity.cache_key("house/bf16", "c4", 10000, 42) != base
    assert fidelity.cache_key("house/bf16", "wikitext103", 20000, 42) != base
    assert fidelity.cache_key("house/bf16", "wikitext103", 10000, 7) != base
    assert fidelity.cache_key("house/bf16", "wikitext103", 10000, 42) == base
    # A model id carries slashes and the key is a file name.
    assert "/" not in fidelity.cache_key("mlx-community/Qwen3-30B", "wikitext103", 10, 1)
