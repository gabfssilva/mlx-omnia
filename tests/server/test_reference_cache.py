"""The reference cache's index and its lifecycle. The pass that fills it is task 59.9's
`# TODO`; what is under test is everything around it, because that is the part the routes
have to be honest about — a cache nobody can see is rubbish accumulating in silence."""

import asyncio
import sys
from collections.abc import Coroutine, Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.server.db import base as db
from mlx_omnia.server.main import migrate
from mlx_omnia.server.services.benchmarks.references import (
    build_reference,
    existing_reference,
    reference,
    reference_key,
    reference_path,
    references,
    save_reference,
)

from .conftest import wired

CACHE_MODULE = sys.modules["mlx_omnia.server.services.benchmarks.references"]
"""Reached through `sys.modules` because the package re-exports `references` the *function*
under that name, so the submodule is not an attribute of it."""


def _loader(model_id: str) -> LanguageModel[ModelInput]:
    raise AssertionError(f"the reference cache loads no model, and asked for {model_id!r}")


def run[T](work: Coroutine[object, object, T]) -> T:
    """One service call against the same file the routes read, with the connection opened and
    closed around it — the lifespan does that for the HTTP half."""

    async def main() -> T:
        await db.connect()
        try:
            return await work
        finally:
            await db.disconnect()

    return asyncio.run(main())


@pytest.fixture
def stand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CACHE_MODULE, "REFERENCE_CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    migrate()


@contextmanager
def served() -> Generator[TestClient]:
    """The routes, over the app's own connection. Opened around the HTTP half only: the
    lifespan owns `db.database` while it runs, and `run` opens the same one."""
    with TestClient(wired(_loader)) as opened:
        yield opened


def _entry(name: str = "house/bf16", corpus: str = "wikitext103") -> str:
    path = reference_path(name, corpus, 10000, 42)
    path.write_bytes(b"\0" * 5_120_000)
    return run(save_reference(name, corpus, 10000, 42, path)).id


def test_the_listing_says_what_the_cache_costs_on_disk(stand: None) -> None:
    _entry("house/bf16")
    _entry("house/other")

    with served() as client:
        body = client.get("/admin/benchmarks/references").json()

    assert len(body["entries"]) == 2
    assert body["total_bytes"] == 2 * 5_120_000
    assert all(entry["present"] for entry in body["entries"])
    assert body["entries"][0]["topk"] == 64


def test_deleting_takes_the_row_and_the_file(stand: None) -> None:
    entry_id = _entry()
    path = reference_path("house/bf16", "wikitext103", 10000, 42)
    assert path.is_file()

    with served() as client:
        assert client.delete(f"/admin/benchmarks/references/{entry_id}").status_code == 204
        assert client.delete(f"/admin/benchmarks/references/{entry_id}").status_code == 404

    assert not path.exists()
    assert run(reference(entry_id)) is None


def test_a_row_whose_file_is_gone_is_not_a_cache_hit(stand: None) -> None:
    """The disk is the truth. The row is dropped so the next pass rebuilds it instead of
    reading a path that is not there."""
    _entry()
    reference_path("house/bf16", "wikitext103", 10000, 42).unlink()

    with served() as client:
        assert client.get("/admin/benchmarks/references").json()["total_bytes"] == 0
    assert run(existing_reference("house/bf16", "wikitext103", 10000, 42)) is None
    assert run(references()) == []


def test_an_entry_already_on_disk_is_reused_instead_of_rebuilt(stand: None) -> None:
    """Which is the whole point: one expensive pass, then five cheap comparisons."""
    entry_id = _entry()

    assert run(build_reference("house/bf16", "wikitext103", 10000, 42)).id == entry_id


def test_building_a_reference_that_is_not_cached_says_which_task_fills_it(stand: None) -> None:
    with pytest.raises(NotImplementedError, match=r"59\.9"):
        run(build_reference("house/bf16", "wikitext103", 10000, 42))


def test_the_key_is_the_four_things_that_change_the_answer(stand: None) -> None:
    """Change any of them and the cached logits answer a different question."""
    base = reference_key("house/bf16", "wikitext103", 10000, 42)

    assert reference_key("house/other", "wikitext103", 10000, 42) != base
    assert reference_key("house/bf16", "c4", 10000, 42) != base
    assert reference_key("house/bf16", "wikitext103", 20000, 42) != base
    assert reference_key("house/bf16", "wikitext103", 10000, 7) != base
    assert reference_key("house/bf16", "wikitext103", 10000, 42) == base
    # A model id carries slashes and the key is a file name.
    assert "/" not in reference_key("mlx-community/Qwen3-30B", "wikitext103", 10, 1)
