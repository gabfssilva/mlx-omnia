"""The database file's three promises: the migration reaches the head from any released
version, the file sits where a user's own data belongs, and what was written is in it.

The flat `Store` is gone — the schema is Alembic's, the domain reads and writes are the
async services, and the engine's two thread-side reads are `db.sync_reads`. What survives
here is what was never about that class: the file itself.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_omnia import LanguageModel, ModelInput, paths
from mlx_omnia.server.db import sync_reads
from mlx_omnia.server.main import migrate

from .conftest import seed_config, wired

_LADDER_VERSION = 11
"""What the flat store's ladder stamped at this schema, and what the initial revision stamps
so a process still running that ladder finds nothing to do."""


def user_version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert isinstance(version, int)
    return version


def table_names(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {name for (name,) in rows}


def journal_mode(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert isinstance(mode, str)
    return mode


def _never(model_id: str) -> LanguageModel[ModelInput]:
    raise AssertionError(f"loading {model_id!r}: nothing here reaches the engine")


def test_the_file_is_left_in_wal() -> None:
    """WAL buys the reader not blocking the writer, which is the daemon's actual shape: a
    long SELECT must not stall the job worker. `db.connect()` sets it, and the pragma is
    persistent — so the file carries it after the app that opened it is gone."""
    with TestClient(wired(_never)):
        pass

    assert journal_mode(paths.server_db()) == "wal"


def test_the_migration_creates_the_whole_schema_over_an_empty_file() -> None:
    path = paths.server_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    migrate()

    assert user_version(path) == _LADDER_VERSION
    assert {"config", "profiles", "jobs"} <= table_names(path)


def test_the_first_boot_creates_the_directory() -> None:
    """`fresh_state` removed the state directory outright: the first migration is what puts
    it back, and a daemon that raised here would never come up on a new machine."""
    assert not paths.state_dir().exists() or not paths.server_db().exists()

    migrate()

    assert paths.server_db().exists()


def test_opening_a_database_already_at_the_head_keeps_its_rows() -> None:
    """Every boot runs the migration. The second one has nothing to do, and doing nothing
    includes not dropping what the first one wrote."""
    seed_config({"port": 8642})

    migrate()

    assert user_version(paths.server_db()) == _LADDER_VERSION
    assert sync_reads.config_values() == {"port": "8642"}


def test_a_database_the_flat_store_built_is_left_exactly_as_it_is() -> None:
    """The upgrade path off the last release: that ladder ended at this schema, so the
    initial revision has nothing to add — and running its DDL would fail on the first table.
    A row written before the move has to still be there afterwards."""
    seed_config({"port": 9042})
    path = paths.server_db()
    with closing(sqlite3.connect(path)) as connection, connection:
        # Alembic's own bookkeeping removed: this is the file as the ladder left it.
        connection.execute("DROP TABLE IF EXISTS alembic_version")

    migrate()

    assert "alembic_version" in table_names(path)
    assert user_version(path) == _LADDER_VERSION
    assert sync_reads.config_values() == {"port": "9042"}
    assert {"config", "profiles", "jobs", "benchmark_runs"} <= table_names(path)


def test_the_benchmark_and_prefix_tables_are_part_of_the_head() -> None:
    """What the last three steps of the ladder added, and what a fresh file gets in one go.
    `benches` was the name before the runs table was split from its bodies."""
    migrate()

    names = table_names(paths.server_db())

    assert {"benchmark_runs", "benchmark_speed", "benchmark_datasets"} <= names
    assert "prefix_cache" in names
    assert "benches" not in names


def test_what_was_written_is_read_back_after_reopening_the_file() -> None:
    """Two apps over one file, sharing nothing but the path."""
    with TestClient(wired(_never)) as first:
        written = first.patch(
            "/admin/config", json={"port": 8642, "memory_limit_bytes": 120_000_000_000}
        )
        assert written.status_code == 200, written.text

    assert sync_reads.config_values() == {
        "port": "8642",
        "memory_limit_bytes": "120000000000",
    }


def test_the_database_sits_where_macos_keeps_what_a_user_backs_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Benchmark history and job rows are the user's own measurements, so Application Support
    and not the log directory. `OMNIA_STATE_DIR` is the override, and what tests point at a
    temp path so a run never touches the real database."""
    monkeypatch.delenv("OMNIA_STATE_DIR", raising=False)
    assert paths.server_db().parent == Path.home() / "Library" / "Application Support" / "mlx-omnia"

    monkeypatch.setenv("OMNIA_STATE_DIR", str(tmp_path))
    assert paths.server_db() == tmp_path / "server.db"
