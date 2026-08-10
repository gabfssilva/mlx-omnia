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
from collections.abc import Callable, Generator, Mapping
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
    features: str = "{}"
    """The engine's own switches, as JSON text — what the profile changes about the
    model's settings, and `{}` when it changes nothing."""


@dataclass(frozen=True)
class ModelSettings:
    """What every request for a model gets before any profile opines. Only features so
    far: sampling defaults already have a home the checkpoint itself writes
    (`generation_config.json`), and a second one would be a second answer."""

    model: str
    features: str = "{}"


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
    subject: str
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


BenchmarkKind = Literal["speed", "fidelity", "quality"]
RunState = Literal["ok", "not_run", "error"]
PageCache = Literal["warm", "cold"]
StreamSource = Literal["queue", "engine"]
Scoring = Literal["loglikelihood", "generate"]
DatasetUse = Literal["multiple_choice", "generation", "corpus"]


@dataclass(frozen=True)
class Run:
    """The header every measurement has, whatever kind it is. `key` is what authorises the
    comparison — it describes the shape of the measurement and never names a model, so the
    same key on two models is one question asked of two candidates — and `(model, key)` is
    the row. Nothing is ever overwritten: two runs of one key under two engine versions are
    the answer to "did Tuesday's optimisation pay"."""

    id: str
    kind: BenchmarkKind
    model: str
    key: str
    state: RunState
    engine_version: str
    mlx_version: str
    created_at: float
    reason: str | None = None
    """Why it did not run, when `state` is `not_run`: `kv_over_budget`,
    `vocabulary_mismatch`, `not_implemented`."""
    temp_c_start: float | None = None
    residents: str = "[]"
    """The other checkpoints holding memory while this was measured, as JSON text. A number
    taken with 60 GB of somebody else's weights resident is not the same number."""


@dataclass(frozen=True)
class SpeedResult:
    """The REAL columns are all optional because `state = 'not_run'` writes this row too:
    what it carries then is the shape that did not fit, which is the whole point of keeping
    it. `step_weight_bytes` and `step_kv_bytes` are stored apart from `ceiling_tps` so the
    fraction can be checked instead of believed."""

    context: int
    generate: int
    concurrency: int
    rounds: int
    warmup: int
    stream_source: StreamSource
    page_cache: PageCache
    gate_c: float | None = None
    load_s: float | None = None
    prefill_tps: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    decode_tps: float | None = None
    decode_per_stream_tps: float | None = None
    step_weight_bytes: int | None = None
    step_kv_bytes: int | None = None
    ceiling_tps: float | None = None
    ceiling_fraction: float | None = None
    per_round: str = "[]"
    """One entry per kept round, as JSON text: the sparkline is the spread, and a median
    with no spread under it hides a thermal ramp."""


@dataclass(frozen=True)
class QualityResult:
    dataset: str
    items: int
    seed: int
    shots: int
    scoring: Scoring
    accuracy: float | None = None
    correct: int | None = None
    wall_s: float | None = None
    breakdown: str = "{}"
    """Accuracy by the dataset's own grouping column, as JSON text."""


@dataclass(frozen=True)
class FidelityResult:
    """Logits, not answers: how far this checkpoint drifted from the one it was made out
    of. `reference` is inside the key, because changing it changes the question."""

    reference: str
    corpus: str
    tokens: int
    seed: int
    topk: int
    kl_mean: float | None = None
    kl_p95: float | None = None
    top1: float | None = None
    top5: float | None = None
    ppl_delta: float | None = None
    histogram: str = "[]"


Result = SpeedResult | QualityResult | FidelityResult


@dataclass(frozen=True)
class BenchmarkRun:
    """What a view hands back: the header and the body of one measurement, together. The
    reader never learns there are two tables."""

    run: Run
    result: Result


@dataclass(frozen=True)
class Dataset:
    """A dataset is data, not code: MMLU and GSM8K are rows of this table seeded by the
    migration, not branches in a scoring function. `columns` maps the roles the template
    names onto the columns the repository actually ships, as JSON text."""

    id: str
    use: DatasetUse
    repo: str
    split: str
    template: str
    created_at: float
    config: str | None = None
    columns: str = "{}"
    size: int | None = None
    builtin: bool = False


@dataclass(frozen=True)
class ReferenceCache:
    """One teacher-forced pass over a corpus, kept as top-k logits plus the logsumexp so
    every candidate after the first is cheap. The weights live in a file — a few megabytes
    per reference does not break sqlite, but it does bloat the one file a user backs up —
    and `path` is what `DELETE` unlinks along with the row."""

    id: str
    reference: str
    corpus: str
    tokens: int
    seed: int
    topk: int
    path: str
    bytes: int
    created_at: float


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

# What the job is about, beside what kind of job it is. The rows already in the file cannot
# answer it — a finished download never wrote down which repository it was — and a row whose
# subject will not parse is a listing that raises forever after. They are dropped: every one
# of them is terminal history, `DELETE` costs the recent-jobs list once, and the alternative
# is a `jobs` endpoint that answers 500 on a file that has been through an upgrade.
_SCHEMA_V3 = """
DELETE FROM jobs;
ALTER TABLE jobs ADD COLUMN subject TEXT NOT NULL DEFAULT '{}';
"""

# A benchmark run is a header plus one body. The alternative — one wide table with a column
# per kind — makes `NOT NULL` unwritable: a speed row has no `accuracy` and a quality row has
# no `decode_tps`, so every column would be nullable and the schema would stop saying which
# ones a kind actually has. The views hand the reader a flat row back, so nothing outside
# this file has to know the pair exists.
#
# `state = 'not_run'` writes the body too, with the REAL columns empty: "128k by 16 did not
# fit" only means something next to the context, the generate and the concurrency it did not
# fit at.
#
# The `benches` table of 34.4 stays as it is. It is a published route, and backfilling these
# tables from it would mean inventing the context, generate and concurrency those
# measurements never had.
_SCHEMA_V4 = """
CREATE TABLE benchmark_runs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('speed', 'fidelity', 'quality')),
    model          TEXT NOT NULL,
    key            TEXT NOT NULL,
    state          TEXT NOT NULL CHECK (state IN ('ok', 'not_run', 'error')),
    reason         TEXT,
    engine_version TEXT NOT NULL,
    mlx_version    TEXT NOT NULL,
    temp_c_start   REAL,
    residents      TEXT NOT NULL DEFAULT '[]',
    created_at     REAL NOT NULL
) STRICT;

CREATE TABLE benchmark_speed (
    run_id                TEXT PRIMARY KEY REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    context               INTEGER NOT NULL,
    generate              INTEGER NOT NULL,
    concurrency           INTEGER NOT NULL,
    rounds                INTEGER NOT NULL,
    warmup                INTEGER NOT NULL,
    stream_source         TEXT NOT NULL CHECK (stream_source IN ('queue', 'engine')),
    page_cache            TEXT NOT NULL CHECK (page_cache IN ('warm', 'cold')),
    gate_c                REAL,
    load_s                REAL,
    prefill_tps           REAL,
    ttft_p50_ms           REAL,
    ttft_p95_ms           REAL,
    decode_tps            REAL,
    decode_per_stream_tps REAL,
    step_weight_bytes     INTEGER,
    step_kv_bytes         INTEGER,
    ceiling_tps           REAL,
    ceiling_fraction      REAL,
    per_round             TEXT NOT NULL DEFAULT '[]'
) STRICT;

CREATE TABLE benchmark_quality (
    run_id    TEXT PRIMARY KEY REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    dataset   TEXT NOT NULL,
    items     INTEGER NOT NULL,
    seed      INTEGER NOT NULL,
    shots     INTEGER NOT NULL,
    scoring   TEXT NOT NULL CHECK (scoring IN ('loglikelihood', 'generate')),
    accuracy  REAL,
    correct   INTEGER,
    wall_s    REAL,
    breakdown TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE TABLE benchmark_fidelity (
    run_id    TEXT PRIMARY KEY REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    reference TEXT NOT NULL,
    corpus    TEXT NOT NULL,
    tokens    INTEGER NOT NULL,
    seed      INTEGER NOT NULL,
    topk      INTEGER NOT NULL,
    kl_mean   REAL,
    kl_p95    REAL,
    top1      REAL,
    top5      REAL,
    ppl_delta REAL,
    histogram TEXT NOT NULL DEFAULT '[]'
) STRICT;

CREATE TABLE benchmark_datasets (
    id         TEXT PRIMARY KEY,
    use        TEXT NOT NULL CHECK (use IN ('multiple_choice', 'generation', 'corpus')),
    repo       TEXT NOT NULL,
    config     TEXT,
    split      TEXT NOT NULL,
    columns    TEXT NOT NULL DEFAULT '{}',
    template   TEXT NOT NULL,
    size       INTEGER,
    builtin    INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
) STRICT;

CREATE TABLE benchmark_reference_cache (
    id         TEXT PRIMARY KEY,
    reference  TEXT NOT NULL,
    corpus     TEXT NOT NULL,
    tokens     INTEGER NOT NULL,
    seed       INTEGER NOT NULL,
    topk       INTEGER NOT NULL,
    path       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at REAL NOT NULL
) STRICT;

CREATE INDEX benchmark_runs_key ON benchmark_runs(kind, key, created_at DESC);
CREATE INDEX benchmark_runs_model ON benchmark_runs(model, created_at DESC);
CREATE INDEX benchmark_speed_shape ON benchmark_speed(context, generate, concurrency);
CREATE INDEX benchmark_fidelity_ref ON benchmark_fidelity(reference);

CREATE VIEW speed_runs AS
SELECT r.id, r.kind, r.model, r.key, r.state, r.reason, r.engine_version, r.mlx_version,
       r.temp_c_start, r.residents, r.created_at,
       s.context, s.generate, s.concurrency, s.rounds, s.warmup, s.stream_source,
       s.page_cache, s.gate_c, s.load_s, s.prefill_tps, s.ttft_p50_ms, s.ttft_p95_ms,
       s.decode_tps, s.decode_per_stream_tps, s.step_weight_bytes, s.step_kv_bytes,
       s.ceiling_tps, s.ceiling_fraction, s.per_round
FROM benchmark_runs r JOIN benchmark_speed s ON s.run_id = r.id;

CREATE VIEW quality_runs AS
SELECT r.id, r.kind, r.model, r.key, r.state, r.reason, r.engine_version, r.mlx_version,
       r.temp_c_start, r.residents, r.created_at,
       q.dataset, q.items, q.seed, q.shots, q.scoring, q.accuracy, q.correct, q.wall_s,
       q.breakdown
FROM benchmark_runs r JOIN benchmark_quality q ON q.run_id = r.id;

CREATE VIEW fidelity_runs AS
SELECT r.id, r.kind, r.model, r.key, r.state, r.reason, r.engine_version, r.mlx_version,
       r.temp_c_start, r.residents, r.created_at,
       f.reference, f.corpus, f.tokens, f.seed, f.topk, f.kl_mean, f.kl_p95, f.top1, f.top5,
       f.ppl_delta, f.histogram
FROM benchmark_runs r JOIN benchmark_fidelity f ON f.run_id = r.id;

INSERT INTO benchmark_datasets
  (id, use, repo, config, split, columns, template, size, builtin, created_at)
VALUES
  ('mmlu', 'multiple_choice', 'cais/mmlu', 'all', 'test',
   '{"question": "question", "choices": "choices", "answer": "answer", "group": "subject"}',
   '{question}\nAnswer:', 14042, 1, 0.0),
  ('arc-challenge', 'multiple_choice', 'allenai/ai2_arc', 'ARC-Challenge', 'test',
   '{"question": "question", "choices": "choices.text", "answer": "answerKey"}',
   '{question}\nAnswer:', 1172, 1, 0.0),
  ('hellaswag', 'multiple_choice', 'Rowan/hellaswag', NULL, 'validation',
   '{"question": "ctx", "choices": "endings", "answer": "label"}',
   '{question}', 10042, 1, 0.0),
  ('gsm8k', 'generation', 'openai/gsm8k', 'main', 'test',
   '{"question": "question", "answer": "answer"}',
   'Question: {question}\nAnswer:', 1319, 1, 0.0),
  ('wikitext103', 'corpus', 'Salesforce/wikitext', 'wikitext-103-raw-v1', 'test',
   '{"text": "text"}', '{text}', NULL, 1, 0.0);
"""

# Entry i takes the schema from version i to i+1. Appending is the only edit allowed:
# a released version is a prefix of this tuple, and that is what makes an old file
# upgradable at all.
# A model's own settings, and the profile's override of them. `features` is JSON text for
# the same reason `sampling` is: what a feature is made of belongs to the feature, and a
# column per one would make every new one a migration. The two levels are separate rows
# because they answer different questions — the model's is what every request gets, the
# profile's is what one preset changes about it — and a profile that says nothing about a
# feature must be distinguishable from one that turns it off.
_SCHEMA_V5 = """
CREATE TABLE model_settings (
    model    TEXT PRIMARY KEY,
    features TEXT NOT NULL DEFAULT '{}'
) STRICT;

ALTER TABLE profiles ADD COLUMN features TEXT NOT NULL DEFAULT '{}';
"""

_MIGRATIONS: tuple[str, ...] = (_SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5)

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
    model, name, sampling, system_prompt, template, features = row
    return Profile(
        model=_text(model),
        name=_text(name),
        sampling=_text(sampling),
        system_prompt=_optional_text(system_prompt),
        template=_optional_text(template),
        features=_text(features),
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
    job_id, kind, subject, state, progress, created_at, updated_at, error = row
    return JobRecord(
        id=_text(job_id),
        kind=_text(kind),
        subject=_text(subject),
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


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _benchmark_kind(value: object) -> BenchmarkKind:
    match _text(value):
        case "speed" | "fidelity" | "quality" as kind:
            return kind
        case other:
            raise ValueError(f"unknown benchmark kind {other!r}")


def _run_state(value: object) -> RunState:
    match _text(value):
        case "ok" | "not_run" | "error" as state:
            return state
        case other:
            raise ValueError(f"unknown run state {other!r}")


def _stream_source(value: object) -> StreamSource:
    match _text(value):
        case "queue" | "engine" as source:
            return source
        case other:
            raise ValueError(f"unknown stream source {other!r}")


def _page_cache(value: object) -> PageCache:
    match _text(value):
        case "warm" | "cold" as cache:
            return cache
        case other:
            raise ValueError(f"unknown page cache state {other!r}")


def _scoring(value: object) -> Scoring:
    match _text(value):
        case "loglikelihood" | "generate" as scoring:
            return scoring
        case other:
            raise ValueError(f"unknown scoring {other!r}")


def _dataset_use(value: object) -> DatasetUse:
    match _text(value):
        case "multiple_choice" | "generation" | "corpus" as use:
            return use
        case other:
            raise ValueError(f"unknown dataset use {other!r}")


def _run(row: tuple[object, ...]) -> Run:
    (
        run_id,
        kind,
        model,
        key,
        state,
        reason,
        engine_version,
        mlx_version,
        temp_c_start,
        residents,
        created_at,
    ) = row
    return Run(
        id=_text(run_id),
        kind=_benchmark_kind(kind),
        model=_text(model),
        key=_text(key),
        state=_run_state(state),
        reason=_optional_text(reason),
        engine_version=_text(engine_version),
        mlx_version=_text(mlx_version),
        temp_c_start=_optional_real(temp_c_start),
        residents=_text(residents),
        created_at=_real(created_at),
    )


def _speed(row: tuple[object, ...]) -> SpeedResult:
    return SpeedResult(
        context=_integer(row[0]),
        generate=_integer(row[1]),
        concurrency=_integer(row[2]),
        rounds=_integer(row[3]),
        warmup=_integer(row[4]),
        stream_source=_stream_source(row[5]),
        page_cache=_page_cache(row[6]),
        gate_c=_optional_real(row[7]),
        load_s=_optional_real(row[8]),
        prefill_tps=_optional_real(row[9]),
        ttft_p50_ms=_optional_real(row[10]),
        ttft_p95_ms=_optional_real(row[11]),
        decode_tps=_optional_real(row[12]),
        decode_per_stream_tps=_optional_real(row[13]),
        step_weight_bytes=_optional_integer(row[14]),
        step_kv_bytes=_optional_integer(row[15]),
        ceiling_tps=_optional_real(row[16]),
        ceiling_fraction=_optional_real(row[17]),
        per_round=_text(row[18]),
    )


def _quality(row: tuple[object, ...]) -> QualityResult:
    return QualityResult(
        dataset=_text(row[0]),
        items=_integer(row[1]),
        seed=_integer(row[2]),
        shots=_integer(row[3]),
        scoring=_scoring(row[4]),
        accuracy=_optional_real(row[5]),
        correct=_optional_integer(row[6]),
        wall_s=_optional_real(row[7]),
        breakdown=_text(row[8]),
    )


def _fidelity(row: tuple[object, ...]) -> FidelityResult:
    return FidelityResult(
        reference=_text(row[0]),
        corpus=_text(row[1]),
        tokens=_integer(row[2]),
        seed=_integer(row[3]),
        topk=_integer(row[4]),
        kl_mean=_optional_real(row[5]),
        kl_p95=_optional_real(row[6]),
        top1=_optional_real(row[7]),
        top5=_optional_real(row[8]),
        ppl_delta=_optional_real(row[9]),
        histogram=_text(row[10]),
    )


def _dataset(row: tuple[object, ...]) -> Dataset:
    dataset_id, use, repo, config, split, columns, template, size, builtin, created_at = row
    return Dataset(
        id=_text(dataset_id),
        use=_dataset_use(use),
        repo=_text(repo),
        config=_optional_text(config),
        split=_text(split),
        columns=_text(columns),
        template=_text(template),
        size=_optional_integer(size),
        builtin=bool(_integer(builtin)),
        created_at=_real(created_at),
    )


def _reference(row: tuple[object, ...]) -> ReferenceCache:
    entry_id, reference, corpus, tokens, seed, topk, path, size, created_at = row
    return ReferenceCache(
        id=_text(entry_id),
        reference=_text(reference),
        corpus=_text(corpus),
        tokens=_integer(tokens),
        seed=_integer(seed),
        topk=_integer(topk),
        path=_text(path),
        bytes=_integer(size),
        created_at=_real(created_at),
    )


_PROFILE_COLUMNS = "model, name, sampling, system_prompt, template, features"
_BENCH_COLUMNS = "model, tokens_per_second, ttft_ms, engine_version, created_at, ceiling_fraction"
_JOB_COLUMNS = "id, kind, subject, state, progress, created_at, updated_at, error"
_SESSION_COLUMNS = "id, title, model, created_at, updated_at, messages"
_RUN_COLUMNS = (
    "id, kind, model, key, state, reason, engine_version, mlx_version, temp_c_start,"
    " residents, created_at"
)
_SPEED_COLUMNS = (
    "context, generate, concurrency, rounds, warmup, stream_source, page_cache, gate_c,"
    " load_s, prefill_tps, ttft_p50_ms, ttft_p95_ms, decode_tps, decode_per_stream_tps,"
    " step_weight_bytes, step_kv_bytes, ceiling_tps, ceiling_fraction, per_round"
)
_QUALITY_COLUMNS = "dataset, items, seed, shots, scoring, accuracy, correct, wall_s, breakdown"
_FIDELITY_COLUMNS = (
    "reference, corpus, tokens, seed, topk, kl_mean, kl_p95, top1, top5, ppl_delta, histogram"
)
_DATASET_COLUMNS = "id, use, repo, config, split, columns, template, size, builtin, created_at"
_REFERENCE_COLUMNS = "id, reference, corpus, tokens, seed, topk, path, bytes, created_at"

_RUN_FIELDS = len(_RUN_COLUMNS.split(","))

# Which view answers for a kind, what its body columns are, and how a row of it is read
# back. Everything about a kind that is not its dataclass lives here, so adding a fourth is
# one entry plus a table.
_BODIES: Mapping[BenchmarkKind, tuple[str, str, str, Callable[[tuple[object, ...]], Result]]] = {
    "speed": ("speed_runs", "benchmark_speed", _SPEED_COLUMNS, _speed),
    "quality": ("quality_runs", "benchmark_quality", _QUALITY_COLUMNS, _quality),
    "fidelity": ("fidelity_runs", "benchmark_fidelity", _FIDELITY_COLUMNS, _fidelity),
}


def _kind_of(result: Result) -> BenchmarkKind:
    match result:
        case SpeedResult():
            return "speed"
        case QualityResult():
            return "quality"
        case FidelityResult():
            return "fidelity"


def _benchmark_run(kind: BenchmarkKind, row: tuple[object, ...]) -> BenchmarkRun:
    _, _, _, read = _BODIES[kind]
    return BenchmarkRun(run=_run(row[:_RUN_FIELDS]), result=read(row[_RUN_FIELDS:]))


class Store:
    """Typed access to the tables, nothing cached: what is read is what is in the file,
    which is what lets a restarted daemon — or a second process — see it."""

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
        # Off by default and per connection, which is the whole trap: without this line the
        # `ON DELETE CASCADE` of the benchmark bodies is decoration, and deleting a run
        # leaves its body orphaned for ever. Outside the transaction below on purpose — the
        # pragma is a no-op inside one.
        connection.execute("PRAGMA foreign_keys = ON")
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
                f"INSERT OR REPLACE INTO profiles ({_PROFILE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    profile.model,
                    profile.name,
                    profile.sampling,
                    profile.system_prompt,
                    profile.template,
                    profile.features,
                ),
            )

    def model_settings(self, model: str) -> ModelSettings:
        """Absent is the same as empty: a model nobody configured has no features set, and
        the caller should not have to tell the two apart."""
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                "SELECT model, features FROM model_settings WHERE model = ?", (model,)
            ).fetchone()
        return ModelSettings(model) if row is None else ModelSettings(_text(row[0]), _text(row[1]))

    def save_model_settings(self, settings: ModelSettings) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO model_settings (model, features) VALUES (?, ?)",
                (settings.model, settings.features),
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
                f"SELECT {_BENCH_COLUMNS} FROM benches{where} ORDER BY created_at DESC, id DESC",
                parameters,
            ).fetchall()
        return [_bench(row) for row in rows]

    def save_job(self, job: JobRecord) -> None:
        """The record is the row: a state transition is the same call as the first write."""
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO jobs ({_JOB_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.kind,
                    job.subject,
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

    def insert_run(self, run: Run, result: Result) -> None:
        """The only door into the benchmark tables, and it takes both halves. Two public
        calls would eventually be used as one — a header written, a body forgotten — and the
        result is a run no view can see. `_connect` already wraps the block in a transaction,
        so the two inserts land together or neither does.

        The kind is taken from the result's own type rather than from the header, so a
        `Run(kind='speed')` carrying a `QualityResult` cannot be written at all.
        """
        kind = _kind_of(result)
        if kind != run.kind:
            raise ValueError(f"{run.kind!r} header carries a {kind!r} result")
        _, table, columns, _ = _BODIES[kind]
        body: tuple[object, ...]
        match result:
            case SpeedResult():
                body = (
                    result.context,
                    result.generate,
                    result.concurrency,
                    result.rounds,
                    result.warmup,
                    result.stream_source,
                    result.page_cache,
                    result.gate_c,
                    result.load_s,
                    result.prefill_tps,
                    result.ttft_p50_ms,
                    result.ttft_p95_ms,
                    result.decode_tps,
                    result.decode_per_stream_tps,
                    result.step_weight_bytes,
                    result.step_kv_bytes,
                    result.ceiling_tps,
                    result.ceiling_fraction,
                    result.per_round,
                )
            case QualityResult():
                body = (
                    result.dataset,
                    result.items,
                    result.seed,
                    result.shots,
                    result.scoring,
                    result.accuracy,
                    result.correct,
                    result.wall_s,
                    result.breakdown,
                )
            case FidelityResult():
                body = (
                    result.reference,
                    result.corpus,
                    result.tokens,
                    result.seed,
                    result.topk,
                    result.kl_mean,
                    result.kl_p95,
                    result.top1,
                    result.top5,
                    result.ppl_delta,
                    result.histogram,
                )
        placeholders = ", ".join("?" * len(body))
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO benchmark_runs ({_RUN_COLUMNS})"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.kind,
                    run.model,
                    run.key,
                    run.state,
                    run.reason,
                    run.engine_version,
                    run.mlx_version,
                    run.temp_c_start,
                    run.residents,
                    run.created_at,
                ),
            )
            connection.execute(
                f"INSERT INTO {table} (run_id, {columns}) VALUES (?, {placeholders})",
                (run.id, *body),
            )

    def runs_by_key(self, kind: BenchmarkKind, key: str) -> list[BenchmarkRun]:
        """One row per model, the most recent — what the comparison band asks: *who has been
        measured under this shape*. The history of a single model is the other direction and
        is `runs_of`."""
        view, _, columns, _ = _BODIES[kind]
        selected = f"{_RUN_COLUMNS}, {columns}"
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {selected} FROM ("
                f" SELECT {selected}, ROW_NUMBER() OVER"
                " (PARTITION BY model ORDER BY created_at DESC, id DESC) AS row_index"
                f" FROM {view} WHERE key = ?"
                ") WHERE row_index = 1 ORDER BY model",
                (key,),
            ).fetchall()
        return [_benchmark_run(kind, row) for row in rows]

    def runs_of(self, kind: BenchmarkKind, model: str) -> list[BenchmarkRun]:
        """Every measurement of one checkpoint, newest first: the series that answers
        whether an engine version moved the number."""
        view, _, columns, _ = _BODIES[kind]
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_RUN_COLUMNS}, {columns} FROM {view} WHERE model = ?"
                " ORDER BY created_at DESC, id DESC",
                (model,),
            ).fetchall()
        return [_benchmark_run(kind, row) for row in rows]

    def runs(self, kind: BenchmarkKind) -> list[BenchmarkRun]:
        view, _, columns, _ = _BODIES[kind]
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_RUN_COLUMNS}, {columns} FROM {view} ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_benchmark_run(kind, row) for row in rows]

    def run(self, run_id: str) -> BenchmarkRun | None:
        """By id, without the caller having to know the kind: the header answers that, and
        the body is read from the matching view."""
        with self._connect() as connection:
            header: tuple[object, ...] | None = connection.execute(
                "SELECT kind FROM benchmark_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if header is None:
                return None
            kind = _benchmark_kind(header[0])
            view, _, columns, _ = _BODIES[kind]
            row: tuple[object, ...] | None = connection.execute(
                f"SELECT {_RUN_COLUMNS}, {columns} FROM {view} WHERE id = ?", (run_id,)
            ).fetchone()
        return None if row is None else _benchmark_run(kind, row)

    def measured(self, kind: BenchmarkKind, model: str, key: str) -> bool:
        """What `skip_if_measured` asks. `not_run` counts as measured: the shape was decided
        against once, and deciding again costs the same arithmetic for the same answer."""
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                "SELECT 1 FROM benchmark_runs WHERE kind = ? AND model = ? AND key = ?"
                " AND state != 'error' LIMIT 1",
                (kind, model, key),
            ).fetchone()
        return row is not None

    def delete_run(self, run_id: str) -> bool:
        """The body goes with it, through the cascade the pragma in `_connect` turns on."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM benchmark_runs WHERE id = ?", (run_id,))
            return cursor.rowcount == 1

    def datasets(self) -> list[Dataset]:
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_DATASET_COLUMNS} FROM benchmark_datasets ORDER BY id"
            ).fetchall()
        return [_dataset(row) for row in rows]

    def dataset(self, dataset_id: str) -> Dataset | None:
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                f"SELECT {_DATASET_COLUMNS} FROM benchmark_datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
        return None if row is None else _dataset(row)

    def save_dataset(self, dataset: Dataset) -> None:
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO benchmark_datasets ({_DATASET_COLUMNS})"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset.id,
                    dataset.use,
                    dataset.repo,
                    dataset.config,
                    dataset.split,
                    dataset.columns,
                    dataset.template,
                    dataset.size,
                    int(dataset.builtin),
                    dataset.created_at,
                ),
            )

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM benchmark_datasets WHERE id = ?", (dataset_id,)
            )
            return cursor.rowcount == 1

    def references(self) -> list[ReferenceCache]:
        with self._connect() as connection:
            rows: list[tuple[object, ...]] = connection.execute(
                f"SELECT {_REFERENCE_COLUMNS} FROM benchmark_reference_cache"
                " ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_reference(row) for row in rows]

    def reference(self, entry_id: str) -> ReferenceCache | None:
        with self._connect() as connection:
            row: tuple[object, ...] | None = connection.execute(
                f"SELECT {_REFERENCE_COLUMNS} FROM benchmark_reference_cache WHERE id = ?",
                (entry_id,),
            ).fetchone()
        return None if row is None else _reference(row)

    def save_reference(self, entry: ReferenceCache) -> None:
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO benchmark_reference_cache ({_REFERENCE_COLUMNS})"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.reference,
                    entry.corpus,
                    entry.tokens,
                    entry.seed,
                    entry.topk,
                    entry.path,
                    entry.bytes,
                    entry.created_at,
                ),
            )

    def delete_reference(self, entry_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM benchmark_reference_cache WHERE id = ?", (entry_id,)
            )
            return cursor.rowcount == 1
