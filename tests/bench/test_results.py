"""The store keyed by the content of `src`, and the report that reads it back. No model and
no GPU: what is under test is where a run lands and what finds it again."""

import json
import subprocess
from pathlib import Path

import pytest

from mlx_omnia.bench import results
from mlx_omnia.bench.results import report, store

PAYLOAD: dict[str, object] = {
    "reference": "omnia",
    "prompt_tokens": 8,
    "comparable": True,
    "arms": {"omnia": {"decode": 100.0, "prefill": 900.0, "ttft": 0.5}},
}


def sh(root: Path, *arguments: str) -> str:
    done = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "engine.py").write_text("WEIGHTS = 1\n")
    (root / "README.md").write_text("readme\n")
    sh(root, "init")
    sh(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    sh(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "one")
    monkeypatch.chdir(root)
    monkeypatch.setattr(results, "CACHE", tmp_path / "cache")
    return root


def stored(root: Path) -> Path:
    return store("interleaved", "fake", {"runs": 2}, PAYLOAD, sustained_gbs=610.0)


def test_a_clean_src_lands_in_the_repo_keyed_by_its_tree(repo: Path) -> None:
    path = stored(repo)

    tree = sh(repo, "rev-parse", "HEAD:src")
    assert path.is_relative_to(repo / "bench/results")
    assert path.parent.name == tree[:12]
    envelope = json.loads(path.read_text())
    assert envelope["tree"] == tree
    assert envelope["dirty"] is False
    machine = json.loads((path.parents[2] / "machine.json").read_text())
    assert machine["sustained_gbs"] == 610.0
    assert path.parents[2].name == machine["slug"]


def test_a_commit_outside_src_still_finds_the_stored_result(repo: Path) -> None:
    stored(repo)
    (repo / "README.md").write_text("changed\n")
    sh(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-am", "docs only")

    rendered = report("HEAD")

    assert "decode   100.0 tok/s" in rendered
    assert sh(repo, "rev-parse", "HEAD:src")[:12] in rendered


def test_a_dirty_src_lands_in_the_cache_and_says_so(repo: Path) -> None:
    (repo / "src" / "engine.py").write_text("WEIGHTS = 2\n")

    path = stored(repo)

    assert path.is_relative_to(results.CACHE)
    assert json.loads(path.read_text())["dirty"] is True
    assert "dirty tree" in report("HEAD")


def test_a_dirty_readme_does_not_evict_a_run_from_the_repo(repo: Path) -> None:
    (repo / "README.md").write_text("changed\n")

    assert stored(repo).is_relative_to(repo / "bench/results")


def test_the_ceiling_line_reads_the_machine_bandwidth(repo: Path) -> None:
    store(
        "interleaved",
        "fake",
        {"runs": 2},
        PAYLOAD,
        sustained_gbs=610.0,
        active_bytes_per_token=6_100_000_000,
    )

    rendered = report("HEAD", model="fake")

    assert "ceiling 100.0 tok/s — omnia at 100.0% of it" in rendered


def test_an_unknown_tree_refuses_with_its_name(repo: Path) -> None:
    with pytest.raises(RuntimeError, match="no stored result"):
        report("a" * 40)


def commit_all(root: Path, message: str) -> None:
    sh(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    sh(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", message)


def test_a_run_survives_commits_to_files_it_never_executed(repo: Path) -> None:
    store(
        "interleaved",
        "fake",
        {"runs": 2},
        PAYLOAD,
        sustained_gbs=610.0,
        executed=[str(repo / "src" / "engine.py")],
    )
    (repo / "src" / "unused.py").write_text("OTHER = 1\n")
    commit_all(repo, "a module the run never imported")

    rendered = report("HEAD")

    assert "decode   100.0 tok/s" in rendered


def test_a_run_dies_with_a_file_it_executed(repo: Path) -> None:
    store(
        "interleaved",
        "fake",
        {"runs": 2},
        PAYLOAD,
        sustained_gbs=610.0,
        executed=[str(repo / "src" / "engine.py")],
    )
    (repo / "src" / "engine.py").write_text("WEIGHTS = 2\n")
    commit_all(repo, "the module the run imported")

    with pytest.raises(RuntimeError, match="no stored result"):
        report("HEAD")


def test_executed_discounts_strategies_that_never_built() -> None:
    """A strategy module is imported to stand in `_STRATEGIES` whether or not it builds,
    so imports alone over-claim; only a resolved strategy counts as executed."""
    import importlib

    from mlx_omnia.bench.arms import omnia as adapter
    from mlx_omnia.engine.core.kernels.resolve import RESOLVED

    name = "mlx_omnia.engine.core.kernels.add_norm.default"
    file = importlib.import_module(name).__file__
    assert file is not None
    saved = set(RESOLVED)
    try:
        RESOLVED.clear()
        assert file not in adapter.executed()
        RESOLVED.add(name)
        assert file in adapter.executed()
    finally:
        RESOLVED.clear()
        RESOLVED.update(saved)


def test_a_run_without_a_file_list_only_matches_its_own_tree(repo: Path) -> None:
    stored(repo)
    (repo / "src" / "unused.py").write_text("OTHER = 1\n")
    commit_all(repo, "src moved, no file list to save the run")

    with pytest.raises(RuntimeError, match="no stored result"):
        report("HEAD")
    assert "decode   100.0 tok/s" in report("HEAD~1")
