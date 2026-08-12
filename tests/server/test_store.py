"""The store's three promises: the migration reaches the head from any released version,
what was written is in the file, and two writers from threads do not lose a row."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from mlx_omnia.server import store
from mlx_omnia.server.store import (
    SCHEMA_VERSION,
    Bench,
    JobRecord,
    Profile,
    QualityResult,
    ReferenceCache,
    Run,
    RunState,
    SpeedResult,
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
    path = tmp_path / "mlx_omnia" / "server.db"

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


def test_the_upgrade_drops_the_job_rows_that_predate_the_subject(tmp_path: Path) -> None:
    """The one step of the ladder that throws something away, on purpose: a row written
    before `subject` existed cannot say what its job was about, and `jobs._subject` refuses
    it — so keeping it would be a listing that raises from the upgrade onwards. What it
    costs is the recent-jobs list, once; what it must not touch is anybody else's table."""
    path = tmp_path / "server.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(store._SCHEMA_V1 + store._SCHEMA_V2)
        connection.execute("PRAGMA user_version = 2")
        connection.execute(
            "INSERT INTO jobs (id, kind, state, progress, created_at, updated_at)"
            " VALUES ('old', 'download', 'ok', '{}', 1.0, 2.0)"
        )
        connection.execute("INSERT INTO config (key, value) VALUES ('port', '8642')")
        connection.commit()

    database = Store(path)

    assert user_version(path) == SCHEMA_VERSION
    assert database.jobs() == []
    assert database.config() == {"port": "8642"}


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
        subject='{"model": "mlx-community/qwen3-30b"}',
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
    subject = '{"model": "a", "target": "b"}'
    failed = JobRecord(
        "job-1", "quantize", subject, "error", '{"step": 2}', 1.0, 9.0, "out of memory"
    )
    database = Store(tmp_path / "server.db")
    database.save_job(JobRecord("job-1", "quantize", subject, "running", '{"step": 1}', 1.0, 1.0))
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
                JobRecord(
                    f"{tag}-{index}",
                    "load",
                    '{"model": "toy"}',
                    "ok",
                    "{}",
                    float(index),
                    float(index),
                )
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

    assert default_path() == tmp_path / "mlx_omnia" / "server.db"


def speed_run(
    run_id: str,
    model: str,
    key: str = "4k→256·1streams·r8+2",
    created_at: float = 1.0,
    engine_version: str = "0.9.2",
    state: RunState = "ok",
    decode_tps: float | None = 100.0,
    reason: str | None = None,
) -> tuple[Run, SpeedResult]:
    header = Run(
        id=run_id,
        kind="speed",
        model=model,
        key=key,
        state=state,
        reason=reason,
        engine_version=engine_version,
        mlx_version="0.30.1",
        created_at=created_at,
    )
    body = SpeedResult(
        context=4096,
        generate=256,
        concurrency=1,
        rounds=8,
        stream_source="queue",
        page_cache="warm",
        decode_tps=decode_tps,
    )
    return header, body


def test_the_benchmark_migration_takes_a_file_at_the_previous_head(tmp_path: Path) -> None:
    """A database the last release left at v3 reaches the head with its rows intact."""
    path = tmp_path / "server.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(store._SCHEMA_V1 + store._SCHEMA_V2 + store._SCHEMA_V3)
        connection.execute("PRAGMA user_version = 3")
        connection.execute("INSERT INTO config (key, value) VALUES ('port', '8642')")
        connection.commit()

    database = Store(path)

    assert user_version(path) == SCHEMA_VERSION
    assert SCHEMA_VERSION == 7
    assert database.config() == {"port": "8642"}
    assert {"benchmark_runs", "benchmark_speed", "benchmark_datasets"} <= table_names(path)
    assert "prefix_cache" in table_names(path)


def test_the_features_migration_leaves_a_profile_written_before_it_unchanged(
    tmp_path: Path,
) -> None:
    """A profile saved at v4 has no `features` column; after the migration it has one, and
    what it says is that the profile opines on nothing — not that it turns anything off."""
    path = tmp_path / "server.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            store._SCHEMA_V1 + store._SCHEMA_V2 + store._SCHEMA_V3 + store._SCHEMA_V4
        )
        connection.execute("PRAGMA user_version = 4")
        connection.execute(
            "INSERT INTO profiles (model, name, sampling) VALUES ('m', 'code', '{}')"
        )
        connection.commit()

    database = Store(path)

    profile = database.profile("m", "code")
    assert profile is not None
    assert profile.features == "{}"
    assert database.model_settings("m").features == "{}"


def test_the_warmup_migration_keeps_the_speed_rows_written_before_it(tmp_path: Path) -> None:
    """The warm-up stopped being part of the shape, so the column goes; a measurement taken
    under v5 still reads back, and it reads back through the view that was rebuilt."""
    path = tmp_path / "server.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            store._SCHEMA_V1
            + store._SCHEMA_V2
            + store._SCHEMA_V3
            + store._SCHEMA_V4
            + store._SCHEMA_V5
        )
        connection.execute("PRAGMA user_version = 5")
        connection.execute(
            "INSERT INTO benchmark_runs (id, kind, model, key, state, engine_version,"
            " mlx_version, created_at) VALUES"
            " ('one', 'speed', 'm', '4k→256 · 1 streams · r3+1 · greedy · warm · queue',"
            " 'ok', '0.9.2', '0.30.0', 1.0)"
        )
        connection.execute(
            "INSERT INTO benchmark_speed (run_id, context, generate, concurrency, rounds,"
            " warmup, stream_source, page_cache, decode_tps) VALUES"
            " ('one', 4096, 256, 1, 3, 1, 'queue', 'warm', 101.0)"
        )
        connection.commit()

    database = Store(path)

    entry = database.run("one")
    assert entry is not None
    assert isinstance(entry.result, SpeedResult)
    assert (entry.result.rounds, entry.result.decode_tps) == (3, 101.0)
    with closing(sqlite3.connect(path)) as connection:
        columns = connection.execute("SELECT * FROM benchmark_speed").description
    assert "warmup" not in {column[0] for column in columns}


def test_deleting_a_run_takes_its_body_with_it(tmp_path: Path) -> None:
    """The cascade only runs because `_connect` turns `foreign_keys` on: with the pragma
    removed this test finds the body still there."""
    path = tmp_path / "server.db"
    database = Store(path)
    database.insert_run(*speed_run("one", "qwen3-30b"))

    assert database.delete_run("one") is True
    assert database.run("one") is None
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT COUNT(*) FROM benchmark_speed").fetchone()
    assert rows == (0,)


def test_a_body_that_does_not_match_its_header_leaves_no_row(tmp_path: Path) -> None:
    """One door, one transaction: a write that fails writes neither half."""
    database = Store(tmp_path / "server.db")
    header, _ = speed_run("one", "qwen3-30b")
    quality = QualityResult(dataset="mmlu", items=1400, seed=42, shots=5, scoring="loglikelihood")

    with pytest.raises(ValueError):
        database.insert_run(header, quality)

    assert database.run("one") is None
    assert database.runs("speed") == []


def test_a_key_answers_with_the_latest_run_of_each_model(tmp_path: Path) -> None:
    """The band asks who was measured under this shape and gets one line per model."""
    database = Store(tmp_path / "server.db")
    database.insert_run(*speed_run("old", "qwen3-30b", created_at=1.0, engine_version="0.9.2"))
    database.insert_run(*speed_run("new", "qwen3-30b", created_at=2.0, engine_version="0.9.3"))
    database.insert_run(*speed_run("other", "gpt-oss-20b", created_at=1.5))
    database.insert_run(*speed_run("elsewhere", "qwen3-30b", key="8k→256·1streams·r8+2"))

    found = database.runs_by_key("speed", "4k→256·1streams·r8+2")

    assert [(entry.run.model, entry.run.engine_version) for entry in found] == [
        ("gpt-oss-20b", "0.9.2"),
        ("qwen3-30b", "0.9.3"),
    ]
    assert len(database.runs_of("speed", "qwen3-30b")) == 3


def test_a_shape_that_did_not_fit_is_a_row_and_not_an_absence(tmp_path: Path) -> None:
    """`not_run` keeps the shape it was decided against, which is what makes the reason
    readable at all."""
    database = Store(tmp_path / "server.db")
    database.insert_run(
        *speed_run("one", "qwen3-30b", state="not_run", reason="kv_over_budget", decode_tps=None)
    )

    (found,) = database.runs_by_key("speed", "4k→256·1streams·r8+2")

    assert found.run.state == "not_run"
    assert found.run.reason == "kv_over_budget"
    assert isinstance(found.result, SpeedResult)
    assert (found.result.context, found.result.concurrency) == (4096, 1)
    assert found.result.decode_tps is None


def test_measured_is_what_skip_if_already_run_asks(tmp_path: Path) -> None:
    database = Store(tmp_path / "server.db")
    database.insert_run(*speed_run("one", "qwen3-30b"))
    database.insert_run(*speed_run("two", "gpt-oss-20b", state="error"))

    assert database.measured("speed", "qwen3-30b", "4k→256·1streams·r8+2") is True
    assert database.measured("speed", "gpt-oss-20b", "4k→256·1streams·r8+2") is False
    assert database.measured("speed", "unknown", "4k→256·1streams·r8+2") is False


def test_the_preset_datasets_are_rows_of_the_table(tmp_path: Path) -> None:
    """MMLU is data, not a branch in a scoring function."""
    database = Store(tmp_path / "server.db")

    found = {dataset.id: dataset for dataset in database.datasets()}

    assert {"mmlu", "arc-challenge", "hellaswag", "gsm8k", "wikitext103"} <= set(found)
    assert found["mmlu"].builtin is True
    assert found["mmlu"].use == "multiple_choice"
    assert found["wikitext103"].use == "corpus"


def test_a_reference_entry_is_indexed_here_and_stored_on_disk(tmp_path: Path) -> None:
    database = Store(tmp_path / "server.db")
    entry = ReferenceCache(
        id="ref-1",
        reference="qwen3-30b-bf16",
        corpus="wikitext103",
        tokens=10000,
        seed=42,
        topk=64,
        path=str(tmp_path / "ref-1.npz"),
        bytes=5_120_000,
        created_at=1.0,
    )
    database.save_reference(entry)

    assert database.references() == [entry]
    assert database.delete_reference("ref-1") is True
    assert database.reference("ref-1") is None
