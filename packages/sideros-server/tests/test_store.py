"""The store's three promises: the migration reaches the head from any released version,
what was written is in the file, and two writers from threads do not lose a row."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from sideros_server import store
from sideros_server.store import (
    SCHEMA_VERSION,
    Bench,
    JobRecord,
    Profile,
    Store,
    default_path,
    migrate,
)


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


def test_the_file_is_left_in_wal(tmp_path: Path) -> None:
    """The concurrency test below passes in rollback mode too — the busy timeout alone
    carries it. What WAL buys is the reader not blocking the writer, which is the daemon's
    actual shape: a long SELECT must not stall the job worker."""
    path = tmp_path / "server.db"

    Store(path)

    assert journal_mode(path) == "wal"


def test_the_migration_creates_the_whole_schema_over_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "server.db"
    path.touch()

    Store(path)

    assert user_version(path) == SCHEMA_VERSION
    assert {"config", "profiles", "benches", "jobs"} <= table_names(path)


def test_the_first_boot_creates_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "sideros" / "server.db"

    Store(path)

    assert path.exists()


def test_opening_a_database_already_at_the_head_keeps_its_rows(tmp_path: Path) -> None:
    """Every boot runs the migration. The second one has nothing to do, and doing nothing
    includes not dropping what the first one wrote."""
    path = tmp_path / "server.db"
    Store(path).set_config({"port": "8642"})

    reopened = Store(path)

    assert user_version(path) == SCHEMA_VERSION
    assert reopened.config() == {"port": "8642"}


def test_the_migration_finishes_a_database_left_at_the_previous_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A released version is a prefix of the ladder, so the file the last release left is
    a database at that prefix: it must reach the head running only the steps it misses,
    with the rows the older schema already held still there. The ladder is stubbed because
    the real one is a single step today and the mechanism has to be exercised anyway."""
    older = "CREATE TABLE toy (id INTEGER PRIMARY KEY, body TEXT NOT NULL) STRICT;"
    newer = "ALTER TABLE toy ADD COLUMN tag TEXT;"
    path = tmp_path / "ladder.db"

    monkeypatch.setattr(store, "_MIGRATIONS", (older,))
    with closing(sqlite3.connect(path)) as connection:
        migrate(connection)
        connection.execute("INSERT INTO toy (id, body) VALUES (1, 'kept')")
        connection.commit()

    monkeypatch.setattr(store, "_MIGRATIONS", (older, newer))
    with closing(sqlite3.connect(path)) as connection:
        migrate(connection)
        row = connection.execute("SELECT body, tag FROM toy").fetchone()

    assert user_version(path) == 2
    assert row == ("kept", None)


def test_what_was_written_is_read_back_after_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "server.db"
    profile = Profile(
        model="qwen3.6-35b",
        name="code",
        sampling='{"temperature": 0.2, "top_p": 0.9}',
        system_prompt="Answer with code and nothing else.",
    )
    bench = Bench(
        model="qwen3.6-35b",
        tokens_per_second=56.6,
        ttft_ms=412.5,
        engine_version="0.9.3",
        created_at=1_753_000_000.0,
        ceiling_fraction=0.566,
    )
    job = JobRecord(
        id="job-1",
        kind="download",
        state="running",
        progress='{"shards": 3}',
        created_at=1.5,
        updated_at=2.5,
    )

    written = Store(path)
    written.set_config({"port": "8642", "memory_limit_bytes": "120000000000"})
    written.save_profile(profile)
    written.add_bench(bench)
    written.save_job(job)

    reopened = Store(path)

    assert reopened.config() == {"port": "8642", "memory_limit_bytes": "120000000000"}
    assert reopened.profile("qwen3.6-35b", "code") == profile
    assert reopened.benches() == [bench]
    assert reopened.job("job-1") == job


def test_a_profile_is_replaced_by_name_and_deleting_says_whether_it_existed(
    tmp_path: Path,
) -> None:
    prose = Profile("qwen3.6-35b", "prose", '{"temperature": 1.1}')
    database = Store(tmp_path / "server.db")
    database.save_profile(Profile("qwen3.6-35b", "code", '{"temperature": 0.2}'))
    database.save_profile(Profile("qwen3.6-35b", "code", '{"temperature": 0.0}'))
    database.save_profile(prose)

    stored = database.profile("qwen3.6-35b", "code")

    assert stored is not None
    assert stored.sampling == '{"temperature": 0.0}'
    assert [profile.name for profile in database.profiles("qwen3.6-35b")] == ["code", "prose"]
    assert database.delete_profile("qwen3.6-35b", "code") is True
    assert database.delete_profile("qwen3.6-35b", "code") is False
    assert database.profiles("qwen3.6-35b") == [prose]


def test_bench_history_comes_back_newest_first_and_filtered_by_model(tmp_path: Path) -> None:
    database = Store(tmp_path / "server.db")
    older = Bench("qwen3.6-35b", 50.1, 402.0, "0.9.2", 10.0, 0.501)
    newer = Bench("qwen3.6-35b", 56.6, 388.0, "0.9.3", 20.0)
    other = Bench("gemma3-4b", 121.0, 90.0, "0.9.3", 30.0, 0.612)
    for bench in (older, newer, other):
        database.add_bench(bench)

    assert database.benches() == [other, newer, older]
    assert database.benches("qwen3.6-35b") == [newer, older]


def test_a_job_is_updated_in_place_by_its_id(tmp_path: Path) -> None:
    failed = JobRecord("job-1", "quantize", "error", '{"step": 2}', 1.0, 9.0, "out of memory")
    database = Store(tmp_path / "server.db")
    database.save_job(JobRecord("job-1", "quantize", "running", '{"step": 1}', 1.0, 1.0))
    database.save_job(failed)

    assert database.jobs() == [failed]


def test_two_stores_writing_from_threads_keep_every_row(tmp_path: Path) -> None:
    """The daemon writes from threads and each call opens its own connection: WAL plus the
    busy timeout is what makes the second writer wait instead of drop its row, and a reader
    running alongside never sees the table shrink."""
    path = tmp_path / "server.db"
    rows = 40
    tags = ("a", "b")
    writers = [Store(path) for _ in tags]
    reader = Store(path)
    counts: list[int] = []
    done = threading.Event()

    def write(writer: Store, tag: str) -> None:
        for index in range(rows):
            writer.save_job(
                JobRecord(f"{tag}-{index}", "toy", "ok", "{}", float(index), float(index))
            )

    def read() -> None:
        while not done.is_set():
            counts.append(len(reader.jobs()))

    with ThreadPoolExecutor(max_workers=len(tags) + 1) as pool:
        watcher = pool.submit(read)
        pairs = zip(writers, tags, strict=True)
        futures = [pool.submit(write, writer, tag) for writer, tag in pairs]
        try:
            for future in futures:
                future.result()
        finally:
            # Without this the reader loops forever on a writer that raised, and the pool's
            # `shutdown(wait=True)` hangs the whole suite instead of reporting the failure.
            done.set()
        watcher.result()

    assert {job.id for job in Store(path).jobs()} == {
        f"{tag}-{index}" for tag in tags for index in range(rows)
    }
    assert counts, "the reader never completed a pass, so the ordering below is vacuous"
    assert counts == sorted(counts)


def test_the_database_sits_beside_the_app_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`XDG_CONFIG_HOME` overrides the `~/.config` base."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert default_path() == tmp_path / "sideros" / "server.db"
