"""The benchmark tables: one header plus one body per measurement, and the reference cache.

A wide table with a column per kind would make `NOT NULL` unwritable — a speed row has no
accuracy, a quality row has no decode rate — so the header is `benchmark_runs` and the body is
one of three tables keyed by its id. `state = 'not_run'` writes the body too, with the real
columns empty: "128k by 16 did not fit" only means something next to the shape it did not fit
at.

`run_id` is the body's primary key and the file carries the cascade on it, so deleting a header
takes its body along.
"""

from __future__ import annotations

from typing import Literal

import ormar

from mlx_omnia.server.db import base

BenchmarkKind = Literal["speed", "fidelity", "quality"]
RunState = Literal["ok", "not_run", "error"]
StreamSource = Literal["queue", "engine"]
PageCache = Literal["warm", "cold"]
Scoring = Literal["loglikelihood", "generate"]

BENCHMARK_KINDS: tuple[BenchmarkKind, ...] = ("speed", "fidelity", "quality")
RUN_STATES: tuple[RunState, ...] = ("ok", "not_run", "error")
STREAM_SOURCES: tuple[StreamSource, ...] = ("queue", "engine")
PAGE_CACHES: tuple[PageCache, ...] = ("warm", "cold")
SCORINGS: tuple[Scoring, ...] = ("loglikelihood", "generate")


class BenchmarkRun(ormar.Model):
    """`key` describes the shape of the measurement and never names a model, so the same key
    on two models is one question asked of two candidates. Nothing is ever overwritten."""

    ormar_config = base.ormar_config.copy(tablename="benchmark_runs")

    id: str = ormar.Text(primary_key=True)
    kind: str = ormar.Text(choices=list(BENCHMARK_KINDS))
    model: str = ormar.Text()
    key: str = ormar.Text()
    state: str = ormar.Text(choices=list(RUN_STATES))
    reason: str | None = ormar.Text(nullable=True)
    engine_version: str = ormar.Text()
    mlx_version: str = ormar.Text()
    temp_c_start: float | None = ormar.Float(nullable=True)
    residents: str = ormar.Text(default="[]")
    created_at: float = ormar.Float()


class SpeedResult(ormar.Model):
    """`step_weight_bytes` and `step_kv_bytes` are stored apart from `ceiling_tps` so the
    fraction can be checked instead of believed."""

    ormar_config = base.ormar_config.copy(tablename="benchmark_speed")

    run_id: str = ormar.Text(primary_key=True)
    context: int = ormar.Integer()
    generate: int = ormar.Integer()
    concurrency: int = ormar.Integer()
    rounds: int = ormar.Integer()
    stream_source: str = ormar.Text(choices=list(STREAM_SOURCES))
    page_cache: str = ormar.Text(choices=list(PAGE_CACHES))
    gate_c: float | None = ormar.Float(nullable=True)
    load_s: float | None = ormar.Float(nullable=True)
    prefill_tps: float | None = ormar.Float(nullable=True)
    ttft_p50_ms: float | None = ormar.Float(nullable=True)
    ttft_p95_ms: float | None = ormar.Float(nullable=True)
    decode_tps: float | None = ormar.Float(nullable=True)
    decode_per_stream_tps: float | None = ormar.Float(nullable=True)
    step_weight_bytes: int | None = ormar.Integer(nullable=True)
    step_kv_bytes: int | None = ormar.Integer(nullable=True)
    ceiling_tps: float | None = ormar.Float(nullable=True)
    ceiling_fraction: float | None = ormar.Float(nullable=True)
    per_round: str = ormar.Text(default="[]")


class QualityResult(ormar.Model):
    ormar_config = base.ormar_config.copy(tablename="benchmark_quality")

    run_id: str = ormar.Text(primary_key=True)
    dataset: str = ormar.Text()
    items: int = ormar.Integer()
    seed: int = ormar.Integer()
    shots: int = ormar.Integer()
    scoring: str = ormar.Text(choices=list(SCORINGS))
    accuracy: float | None = ormar.Float(nullable=True)
    correct: int | None = ormar.Integer(nullable=True)
    wall_s: float | None = ormar.Float(nullable=True)
    breakdown: str = ormar.Text(default="{}")


class FidelityResult(ormar.Model):
    """Logits, not answers: how far this checkpoint drifted from the one it was made out of.
    `reference` is inside the run's key, because changing it changes the question."""

    ormar_config = base.ormar_config.copy(tablename="benchmark_fidelity")

    run_id: str = ormar.Text(primary_key=True)
    reference: str = ormar.Text()
    corpus: str = ormar.Text()
    tokens: int = ormar.Integer()
    seed: int = ormar.Integer()
    topk: int = ormar.Integer()
    kl_mean: float | None = ormar.Float(nullable=True)
    kl_p95: float | None = ormar.Float(nullable=True)
    top1: float | None = ormar.Float(nullable=True)
    top5: float | None = ormar.Float(nullable=True)
    ppl_delta: float | None = ormar.Float(nullable=True)
    histogram: str = ormar.Text(default="[]")


class ReferenceCache(ormar.Model):
    """One teacher-forced pass over a corpus, kept as top-k logits plus the logsumexp. The
    weights live in a file — `path` is what a delete unlinks along with the row."""

    ormar_config = base.ormar_config.copy(tablename="benchmark_reference_cache")

    id: str = ormar.Text(primary_key=True)
    reference: str = ormar.Text()
    corpus: str = ormar.Text()
    tokens: int = ormar.Integer()
    seed: int = ormar.Integer()
    topk: int = ormar.Integer()
    path: str = ormar.Text()
    bytes: int = ormar.Integer()
    created_at: float = ormar.Float()
