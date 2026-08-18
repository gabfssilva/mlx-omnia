"""Initial schema: the head the flat store's migration ladder reached.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# The version the old `store.migrate()` ladder stamped in `PRAGMA user_version` at this schema.
# A file this revision creates is stamped too, so a process still running the ladder finds
# nothing to do instead of replaying it over these tables.
_LADDER_VERSION = 11

_SCHEMA = """
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
    features      TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (model, name)
) STRICT;

CREATE TABLE model_settings (
    model                   TEXT PRIMARY KEY,
    features                TEXT NOT NULL DEFAULT '{}',
    max_concurrent_requests INTEGER,
    sampling                TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE TABLE jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    state      TEXT NOT NULL
               CHECK (state IN ('pending', 'running', 'ok', 'error', 'cancelled')),
    progress   TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error      TEXT,
    subject    TEXT NOT NULL DEFAULT '{}'
) STRICT;

CREATE TABLE sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    messages   TEXT NOT NULL DEFAULT '[]'
) STRICT;

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

CREATE TABLE prefix_cache (
    key        TEXT NOT NULL,
    kind       TEXT NOT NULL,
    model      TEXT NOT NULL,
    path       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at REAL NOT NULL,
    used_at    REAL NOT NULL,
    PRIMARY KEY (key, kind)
) STRICT;

CREATE INDEX benchmark_runs_key ON benchmark_runs(kind, key, created_at DESC);
CREATE INDEX benchmark_runs_model ON benchmark_runs(model, created_at DESC);
CREATE INDEX benchmark_speed_shape ON benchmark_speed(context, generate, concurrency);
CREATE INDEX benchmark_fidelity_ref ON benchmark_fidelity(reference);
CREATE INDEX prefix_cache_lru ON prefix_cache(used_at);
CREATE INDEX prefix_cache_model ON prefix_cache(model);
"""

_BUILTIN_DATASETS: tuple[dict[str, object], ...] = (
    {
        "id": "mmlu",
        "use": "multiple_choice",
        "repo": "cais/mmlu",
        "config": "all",
        "split": "test",
        "columns": (
            '{"question": "question", "choices": "choices", "answer": "answer",'
            ' "group": "subject"}'
        ),
        "template": "{question}\nAnswer:",
        "size": 14042,
        "builtin": 1,
        "created_at": 0.0,
    },
    {
        "id": "arc-challenge",
        "use": "multiple_choice",
        "repo": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "columns": '{"question": "question", "choices": "choices.text", "answer": "answerKey"}',
        "template": "{question}\nAnswer:",
        "size": 1172,
        "builtin": 1,
        "created_at": 0.0,
    },
    {
        "id": "hellaswag",
        "use": "multiple_choice",
        "repo": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
        "columns": '{"question": "ctx", "choices": "endings", "answer": "label"}',
        "template": "{question}",
        "size": 10042,
        "builtin": 1,
        "created_at": 0.0,
    },
    {
        "id": "gsm8k",
        "use": "generation",
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "columns": '{"question": "question", "answer": "answer"}',
        "template": "Question: {question}\nAnswer:",
        "size": 1319,
        "builtin": 1,
        "created_at": 0.0,
    },
    {
        "id": "wikitext103",
        "use": "corpus",
        "repo": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "test",
        "columns": '{"text": "text"}',
        "template": "{text}",
        "size": None,
        "builtin": 1,
        "created_at": 0.0,
    },
)

_SEED = sa.text(
    "INSERT INTO benchmark_datasets"
    " (id, use, repo, config, split, columns, template, size, builtin, created_at)"
    " VALUES (:id, :use, :repo, :config, :split, :columns, :template, :size, :builtin,"
    " :created_at)"
)


def upgrade() -> None:
    """A file the flat store already built is left exactly as it is.

    This revision describes the schema that ladder ended at, so on such a file it has nothing
    to add; running the DDL would fail on the first table. It only builds a fresh file.
    """
    connection = op.get_bind()
    if sa.inspect(connection).has_table("jobs"):
        return
    for statement in filter(None, (part.strip() for part in _SCHEMA.split(";"))):
        connection.execute(sa.text(statement))
    for dataset in _BUILTIN_DATASETS:
        connection.execute(_SEED, dataset)
    connection.execute(sa.text(f"PRAGMA user_version = {_LADDER_VERSION}"))


def downgrade() -> None:
    raise NotImplementedError("the initial revision is the floor")
