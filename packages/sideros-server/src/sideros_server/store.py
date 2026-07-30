"""The state that outlives the process: config, profiles, bench history, recent jobs and
the chat sessions.

Raw `sqlite3` from the stdlib and nothing above it. Each consumer reads and writes whole
rows of one table; an ORM would add a mapping layer to keep in sync with the schema
without removing a single line of the SQL below.

The file is `~/.config/sideros/server.db`, honouring `XDG_CONFIG_HOME` — the directory
the app already writes `app.json` into. Under XDG's reading, bench history and jobs are
state rather than configuration; one directory holding everything a user backs up or
deletes is worth more here than the classification.

Every call opens and closes its own connection. The daemon writes from threads (the job
worker runs under `to_thread`) and a sqlite3 connection does not travel between them;
WAL plus the busy timeout is what makes a second writer — another thread, or a second
process — wait instead of fail.
"""

import os
import sqlite3
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_BUSY_SECONDS = 5.0

JobState = Literal["pending", "running", "ok", "error", "cancelled"]


def default_path() -> Path:
    """`~/.config/sideros/server.db`, honoring `XDG_CONFIG_HOME`."""
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root) if root else Path.home() / ".config"
    return base / "sideros" / "server.db"


@dataclass(frozen=True)
class Profile:
    """`sampling` is JSON text: what a profile may set is the sampling module's
    vocabulary, and a column per knob would make every new one a migration."""

    model: str
    name: str
    sampling: str
    system_prompt: str | None = None
    template: str | None = None


@dataclass(frozen=True)
class Bench:
    """`ceiling_fraction` is absent when the checkpoint's bytes per token — and so the
    bandwidth ceiling the number is a percentage of — is not known."""

    model: str
    tokens_per_second: float
    ttft_ms: float
    engine_version: str
    created_at: float
    ceiling_fraction: float | None = None


@dataclass(frozen=True)
class JobRecord:
    """What is left of a job once its stream is gone: `progress` is the last frame, as
    JSON text, so a client that arrives after a restart still sees where it stopped."""

    id: str
    kind: str
    state: JobState
    progress: str
    created_at: float
    updated_at: float
    error: str | None = None


@dataclass(frozen=True)
class Session:
    """A conversation as the file holds it. `messages` is JSON text and stays opaque here:
    what a message is, is the dialect's vocabulary — content parts, tool calls, reasoning —
    and a table under it would make every new part a migration of somebody else's shape."""

    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    messages: str


@dataclass(frozen=True)
class SessionSummary:
    """What the list answers: the same row without its conversation. The messages are the
    whole weight of the table, and the sidebar wants a count, not a transcript."""

    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    message_count: int


_SCHEMA_V1 = """
CREATE TABLE config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE profiles (
    model         TEXT NOT NULL,
    name          TEXT NOT NULL,
    sampling      TEXT NOT NULL,
    system_prompt TEXT,
    template      TEXT,
    PRIMARY KEY (model, name)
) STRICT;

CREATE TABLE benches (
    id                INTEGER PRIMARY KEY,
    model             TEXT NOT NULL,
    tokens_per_second REAL NOT NULL,
    ttft_ms           REAL NOT NULL,
    engine_version    TEXT NOT NULL,
    created_at        REAL NOT NULL,
    ceiling_fraction  REAL
) STRICT;

CREATE TABLE jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    state      TEXT NOT NULL
               CHECK (state IN ('pending', 'running', 'ok', 'error', 'cancelled')),
    progress   TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error      TEXT
) STRICT;
"""

_SCHEMA_V2 = """
CREATE TABLE sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    messages   TEXT NOT NULL DEFAULT '[]'
) STRICT;
"""

# Entry i takes the schema from version i to i+1. Appending is the only edit allowed:
# a released version is a prefix of this tuple, and that is what makes an old file
# upgradable at all.
_MIGRATIONS: tuple[str, ...] = (_SCHEMA_V1, _SCHEMA_V2)

SCHEMA_VERSION = len(_MIGRATIONS)


def migrate(connection: sqlite3.Connection) -> None:
    """Walks the ladder from wherever this file is to the head.

    `PRAGMA user_version` is the ledger, so an empty file (version 0) and the file last
    release left take the same path: the steps they are missing, in order. Each step
    carries its own transaction — an interrupted upgrade leaves the previous version
    intact instead of half a schema.
    """
    version = _integer(connection.execute("PRAGMA user_version").fetchone()[0])
    for index in range(version, len(_MIGRATIONS)):
        connection.executescript(
            f"BEGIN;\n{_MIGRATIONS[index]}\nPRAGMA user_version = {index + 1};\nCOMMIT;"
        )


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected INTEGER, got {value!r}")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected TEXT, got {value!r}")
    return value


def _real(value: object) -> float:
    if not isinstance(value, float):
        raise TypeError(f"expected REAL, got {value!r}")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _optional_real(value: object) -> float | None:
    return None if value is None else _real(value)


def _job_state(value: object) -> JobState:
    match _text(value):
        case "pending" | "running" | "ok" | "error" | "cancelled" as state:
            return state
        case other:
            raise ValueError(f"unknown job state {other!r}")


def _profile(row: tuple[object, ...]) -> Profile:
    model, name, sampling, system_prompt, template = row
    return Profile(
        model=_text(model),
        name=_text(name),
        sampling=_text(sampling),
        system_prompt=_optional_text(system_prompt),
        template=_optional_text(template),
    )


def _bench(row: tuple[object, ...]) -> Bench:
    model, tokens_per_second, ttft_ms, engine_version, created_at, ceiling_fraction = row
    return Bench(
        model=_text(model),
        tokens_per_second=_real(tokens_per_second),
        ttft_ms=_real(ttft_ms),
        engine_version=_text(engine_version),
        created_at=_real(created_at),
        ceiling_fraction=_optional_real(ceiling_fraction),
    )


def _job(row: tuple[object, ...]) -> JobRecord:
    job_id, kind, state, progress, created_at, updated_at, error = row
    return JobRecord(
        id=_text(job_id),
        kind=_text(kind),
        state=_job_state(state),
        progress=_text(progress),
        created_at=_real(created_at),
        updated_at=_real(updated_at),
        error=_optional_text(error),
    )


def _session(row: tuple[object, ...]) -> Session:
    session_id, title, model, created_at, updated_at, messages = row
    return Session(
        id=_text(session_id),
        title=_text(title),
        model=_text(model),
        created_at=_real(created_at),
        updated_at=_real(updated_at),
        messages=_text(messages),
    )


def _session_summary(row: tuple[object, ...]) -> SessionSummary:
    session_id, title, model, created_at, updated_at, message_count = row
    return SessionSummary(
        id=_text(session_id),
        title=_text(title),
        model=_text(model),
        created_at=_real(created_at),
        updated_at=_real(updated_at),
        message_count=_integer(message_count),
    )


_PROFILE_COLUMNS = "model, name, sampling, system_prompt, template"
_BENCH_COLUMNS = "model, tokens_per_second, ttft_ms, engine_version, created_at, ceiling_fraction"
_JOB_COLUMNS = "id, kind, state, progress, created_at, updated_at, error"
_SESSION_COLUMNS = "id, title, model, created_at, updated_at, messages"


class Store:
    """Typed access to the five tables, nothing cached: what is read is what is in the
    file, which is what lets a restarted daemon — or a second process — see it."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = default_path() if path is None else path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            migrate(connection)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        """One connection per call, in the calling thread, committing on the way out."""
        connection = sqlite3.connect(self._path, timeout=_BUSY_SECONDS)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def config(self) -> dict[str, str]:
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                "SELECT key, value FROM config"
            ).fetchall()
        return {_text(key): _text(value) for key, value in rows}

    def set_config(self, values: Mapping[str, str]) -> None:
        """A PATCH carries several keys: they land together or not at all."""
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", values.items()
            )

    def profiles(self, model: str) -> list[Profile]:
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE model = ? ORDER BY name",
                (model,),
            ).fetchall()
        return [_profile(row) for row in rows]

    def profile_names(self) -> dict[str, list[str]]:
        """Every profile name, by model, in one query. The model list answers with one entry
        per name a checkpoint responds to, and asking per checkpoint is a connection each —
        the catalog is hundreds of entries and the handler runs on the loop."""
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                "SELECT model, name FROM profiles ORDER BY model, name"
            ).fetchall()
        names: dict[str, list[str]] = {}
        for model, name in rows:
            names.setdefault(_text(model), []).append(_text(name))
        return names

    def profile(self, model: str, name: str) -> Profile | None:
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                f"SELECT {_PROFILE_COLUMNS} FROM profiles WHERE model = ? AND name = ?",
                (model, name),
            ).fetchone()
        return None if row is None else _profile(row)

    def save_profile(self, profile: Profile) -> None:
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO profiles ({_PROFILE_COLUMNS}) VALUES (?, ?, ?, ?, ?)",
                (
                    profile.model,
                    profile.name,
                    profile.sampling,
                    profile.system_prompt,
                    profile.template,
                ),
            )

    def delete_profile(self, model: str, name: str) -> bool:
        """Says whether there was one, so the route answers 404 instead of a silent 204."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM profiles WHERE model = ? AND name = ?", (model, name)
            )
            return cursor.rowcount == 1

    def add_bench(self, bench: Bench) -> None:
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO benches ({_BENCH_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    bench.model,
                    bench.tokens_per_second,
                    bench.ttft_ms,
                    bench.engine_version,
                    bench.created_at,
                    bench.ceiling_fraction,
                ),
            )

    def benches(self, model: str | None = None) -> list[Bench]:
        """Newest first: the model card shows the last measurement, the history under it."""
        where = "" if model is None else " WHERE model = ?"
        parameters: tuple[str, ...] = () if model is None else (model,)
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_BENCH_COLUMNS} FROM benches{where}"
                " ORDER BY created_at DESC, id DESC",
                parameters,
            ).fetchall()
        return [_bench(row) for row in rows]

    def save_job(self, job: JobRecord) -> None:
        """The record is the row: a state transition is the same call as the first write."""
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO jobs ({_JOB_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.kind,
                    job.state,
                    job.progress,
                    job.created_at,
                    job.updated_at,
                    job.error,
                ),
            )

    def abandon_jobs(self) -> int:
        """Every job left mid-flight by a process that is gone, marked as what it is, and
        answering how many there were. A row reaches this state only through a kill the
        daemon could not answer — the loop's own shutdown finishes what it is running — and
        nothing else ever reconciles it: the progress screen would open on a bar that never
        moves, and `DELETE` on that job answers 409 because no live one carries its id."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET state = 'error', error = ?, updated_at = ?"
                " WHERE state IN ('pending', 'running')",
                ("the daemon stopped before this job finished", time.time()),
            )
        return cursor.rowcount

    def job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return None if row is None else _job(row)

    def jobs(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_job(row) for row in rows]

    def sessions(self) -> list[SessionSummary]:
        """Most recently touched first, which is the order the sidebar draws. The count is
        `json_array_length` rather than a parse on this side: the conversation is the only
        large column in the table, and the list is the one call that reads every row."""
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                "SELECT id, title, model, created_at, updated_at, json_array_length(messages)"
                " FROM sessions ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [_session_summary(row) for row in rows]

    def session(self, session_id: str) -> Session | None:
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return None if row is None else _session(row)

    def save_session(self, session: Session) -> None:
        """The record is the row, as for jobs: the create, the rename and the turn appended
        to a conversation are the same call, and the caller is what decides `updated_at`."""
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO sessions ({_SESSION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.title,
                    session.model,
                    session.created_at,
                    session.updated_at,
                    session.messages,
                ),
            )

    def delete_session(self, session_id: str) -> bool:
        """Says whether there was one, so the route answers 404 instead of a silent 204."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount == 1
