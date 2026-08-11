"""The batch, its dry run, and the history it writes.

A benchmark is a cartesian product of shapes that runs for minutes, so it is a job in the
sense 34.1 fixed: `202` plus a row, frames over the SSE that is already there, and a
cancellation that reaches inside the work. What is different from every other job here is
that its result is only worth what the machine was doing beside it — so the work holds the
engine's queue for as long as it measures, and everything else waits at `submit` instead of
landing between two rounds and moving the median.

The expansion is written once and used twice. `POST /admin/benchmarks/plan` answers what
would run, what would be skipped and why, and the job runs exactly that: a second copy of
the arithmetic is a second answer to "does 128k by 16 fit", and one of the two would go
stale the first time the cache dtype moved. Which is also why the client never recomputes
it — the sheet draws the plan's numbers, including the `needs 206 GB`.

Speed measures. Quality and fidelity validate their body, expand their shapes and write
their rows as `not_run`, then finish the job in `error` naming the task that fills the
`# TODO`: a job that ends `ok` with an invented number is the one outcome this section
exists to not produce.
"""

import asyncio
import csv
import io
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated, Literal

import mlx.core as mx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia_server import catalog, speed
from mlx_omnia_server.config import current
from mlx_omnia_server.engine import Engine
from mlx_omnia_server.jobs import Benchmark as BenchmarkJob
from mlx_omnia_server.jobs import Cancelled, Job, JobsDep, Progress, Work, accepted
from mlx_omnia_server.profiles import StoreDep
from mlx_omnia_server.speed import ModelFacts, Refusal, SpeedShape
from mlx_omnia_server.store import (
    BenchmarkKind,
    BenchmarkRun,
    FidelityResult,
    QualityResult,
    Result,
    Run,
    RunState,
    SpeedResult,
    Store,
)

_MAX_SHAPES = 512
"""Everything the three sets can produce for one model is 385 shapes; past this the request
is a mistake and saying so beats spending a night on it."""

QUALITY_TODO = (
    "quality scoring is not implemented: the log-likelihood pass over a dataset's"
    " continuations is task 59.8. The shapes were expanded and recorded as not_run."
)

FIDELITY_TODO = (
    "fidelity is not implemented: the teacher-forced pass that produces the reference's"
    " top-k logits, and the comparison that produces KL, top-1, top-5 and Δppl, is task"
    " 59.9. The pairs were expanded and recorded as not_run."
)


class SamplingBody(BaseModel):
    """The sampler the rounds decode under. The bounds are the dialects' own
    (`app.ChatRequest`), and the defaults are the deterministic end of every one of them:
    a benchmark that says nothing about sampling is measuring an argmax."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = None

    def shape(self) -> speed.Sampling:
        return speed.Sampling(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )


class SpeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["speed"] = "speed"
    models: list[str] = Field(min_length=1)
    contexts: list[int] = Field(min_length=1)
    generates: list[int] = Field(min_length=1)
    concurrencies: list[int] = Field(min_length=1)
    rounds: int = Field(default=3, ge=1, le=64)
    sampling: SamplingBody = SamplingBody()
    page_cache: Literal["warm", "cold"] = "warm"
    thermal_gate_c: float | None = Field(default=None, ge=35, le=90)
    stream_source: Literal["queue", "engine"] = "queue"
    skip_if_measured: bool = True


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reference: str


class FidelityRequest(BaseModel):
    """Not a multi-selection: each candidate names the reference it is measured against,
    because the reference is inside the key and changing it changes the question."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["fidelity"]
    pairs: list[Pair] = Field(min_length=1)
    corpus: str = "wikitext103"
    tokens: int = Field(default=10000, ge=128)
    seed: int = 42
    topk: int = Field(default=64, ge=1, le=1024)
    skip_if_measured: bool = True


class QualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["quality"]
    models: list[str] = Field(min_length=1)
    datasets: list[str] = Field(min_length=1)
    items: int = Field(default=1400, ge=1)
    seed: int = 42
    shots: int = Field(default=5, ge=0, le=64)
    scoring: Literal["loglikelihood", "generate"] = "loglikelihood"
    skip_if_measured: bool = True


BenchmarkRequest = Annotated[
    SpeedRequest | FidelityRequest | QualityRequest, Field(discriminator="kind")
]


@dataclass(frozen=True)
class Planned:
    """One row the batch will write. `shape` is present only where something can actually
    run today; `refusal` is present when the row is already decided."""

    model: str
    key: str
    kind: BenchmarkKind
    body: Result
    shape: SpeedShape | None = None
    refusal: Refusal | None = None
    warning: str | None = None
    """Something the sheet has to say beside the row without refusing it."""


def quality_key(dataset: str, items: int, seed: int, shots: int) -> str:
    return f"{dataset} · n{items} · s{seed} · {shots}shot"


def fidelity_key(corpus: str, tokens: int, seed: int, reference: str) -> str:
    return f"{corpus} · n{tokens} · s{seed} · ref:{reference}"


def _facts(entries: Sequence[catalog.CatalogEntry]) -> Mapping[str, ModelFacts]:
    return {entry.id: speed.facts_of(entry) for entry in entries}


def _closed(name: str, asked: Sequence[int], allowed: tuple[int, ...]) -> None:
    unknown = sorted(set(asked) - set(allowed))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"{name}: {unknown} is outside the fixed set {list(allowed)}",
        )


def expand(
    body: SpeedRequest | FidelityRequest | QualityRequest,
    entries: Sequence[catalog.CatalogEntry],
    budget_bytes: int,
) -> list[Planned]:
    """Every row the batch would write, decided ones included. The one place the product is
    taken, so the plan and the job cannot disagree."""
    known = {entry.id: entry for entry in entries}
    match body:
        case SpeedRequest():
            return _expand_speed(body, known, _facts(entries), budget_bytes)
        case QualityRequest():
            return _expand_quality(body, known)
        case FidelityRequest():
            return _expand_fidelity(body, known)


def _missing(model: str, known: Mapping[str, catalog.CatalogEntry]) -> None:
    if model not in known:
        raise HTTPException(status_code=404, detail=f"{model!r} is not in the catalog")


def _expand_speed(
    body: SpeedRequest,
    known: Mapping[str, catalog.CatalogEntry],
    facts: Mapping[str, ModelFacts],
    budget_bytes: int,
) -> list[Planned]:
    _closed("contexts", body.contexts, speed.CONTEXTS)
    _closed("generates", body.generates, speed.GENERATES)
    _closed("concurrencies", body.concurrencies, speed.CONCURRENCIES)
    planned: list[Planned] = []
    for model in body.models:
        _missing(model, known)
        for context in sorted(set(body.contexts)):
            for generate in sorted(set(body.generates)):
                for concurrency in sorted(set(body.concurrencies)):
                    shape = SpeedShape(
                        context=context,
                        generate=generate,
                        concurrency=concurrency,
                        rounds=body.rounds,
                        page_cache=body.page_cache,
                        gate_c=body.thermal_gate_c,
                        stream_source=body.stream_source,
                        sampling=body.sampling.shape(),
                    )
                    refused = speed.refusal(shape, facts[model], budget_bytes)
                    planned.append(
                        Planned(
                            model=model,
                            key=shape.key,
                            kind="speed",
                            body=speed.empty_result(shape, facts[model], refused),
                            shape=shape,
                            refusal=refused,
                        )
                    )
    if len(planned) > _MAX_SHAPES:
        raise HTTPException(
            status_code=400,
            detail=f"{len(planned)} shapes asked for, {_MAX_SHAPES} is the ceiling",
        )
    return planned


def _expand_quality(
    body: QualityRequest, known: Mapping[str, catalog.CatalogEntry]
) -> list[Planned]:
    planned: list[Planned] = []
    for model in body.models:
        _missing(model, known)
        for dataset in body.datasets:
            planned.append(
                Planned(
                    model=model,
                    key=quality_key(dataset, body.items, body.seed, body.shots),
                    kind="quality",
                    body=QualityResult(
                        dataset=dataset,
                        items=body.items,
                        seed=body.seed,
                        shots=body.shots,
                        scoring=body.scoring,
                    ),
                    refusal=Refusal(reason="not_implemented", detail=QUALITY_TODO),
                )
            )
    return planned


def _expand_fidelity(
    body: FidelityRequest, known: Mapping[str, catalog.CatalogEntry]
) -> list[Planned]:
    planned: list[Planned] = []
    for pair in body.pairs:
        _missing(pair.model, known)
        _missing(pair.reference, known)
        candidate, reference = known[pair.model], known[pair.reference]
        # Vocabulary is what decides whether the two vectors can be subtracted at all, and
        # it is decided here rather than by the pass that never runs: a mismatched pair
        # costs nothing to refuse and would cost a load to discover.
        mismatch = (
            candidate.vocab_size is not None
            and reference.vocab_size is not None
            and candidate.vocab_size != reference.vocab_size
        )
        refused = (
            Refusal(
                reason="vocabulary_mismatch",
                detail=(
                    f"{pair.model!r} draws from {candidate.vocab_size} ids and"
                    f" {pair.reference!r} from {reference.vocab_size}: their logits are not"
                    " the same vector"
                ),
            )
            if mismatch
            else Refusal(reason="not_implemented", detail=FIDELITY_TODO)
        )
        # Same vocabulary is necessary and not sufficient: Qwen3-14B and Qwen3-30B-A3B share
        # a tokenizer, so KL between them computes and says nothing this view means. It
        # warns rather than refuses (decision of task 59.9) because the print cannot tell
        # the two cases apart — a fine-tune prints identically to its base, so blocking on a
        # different print would block only the pairs somebody deliberately declared.
        warning = (
            None
            if mismatch or candidate.shape is None or candidate.shape == reference.shape
            else (
                f"{pair.model!r} is {candidate.shape} and {pair.reference!r} is"
                f" {reference.shape}: the vocabularies match, the architectures do not"
            )
        )
        planned.append(
            Planned(
                model=pair.model,
                key=fidelity_key(body.corpus, body.tokens, body.seed, pair.reference),
                kind="fidelity",
                body=FidelityResult(
                    reference=pair.reference,
                    corpus=body.corpus,
                    tokens=body.tokens,
                    seed=body.seed,
                    topk=body.topk,
                ),
                refusal=refused,
                warning=warning,
            )
        )
    return planned


class PlannedShape(BaseModel):
    model: str
    key: str


class SkippedShape(BaseModel):
    model: str
    key: str
    reason: str
    detail: str | None = None
    needed_bytes: int | None = None
    budget_bytes: int | None = None


class Plan(BaseModel):
    """What the sheet draws before anybody presses run. The three reasons for not running
    are separate because the sheet separates them: too big, already answered, and — for
    fidelity — not comparable."""

    shapes: list[PlannedShape]
    skipped: list[SkippedShape]
    already: list[PlannedShape]
    warnings: list[SkippedShape]
    """Rows that will run and that the sheet has to mark — a fidelity pair whose two
    architectures differ is the one case. Separate from `skipped` because a warning is not
    a refusal."""
    estimate_seconds: float | None
    """`null` rather than a guessed number when there is neither history nor a ceiling to
    divide by."""


def _split(
    planned: Sequence[Planned], store: Store, skip_if_measured: bool
) -> tuple[list[Planned], list[Planned], list[Planned]]:
    """Runnable, already answered, refused — in that order, and by one rule shared with the
    job."""
    runnable: list[Planned] = []
    already: list[Planned] = []
    skipped: list[Planned] = []
    for entry in planned:
        if skip_if_measured and store.measured(entry.kind, entry.model, entry.key):
            already.append(entry)
        elif entry.refusal is not None:
            skipped.append(entry)
        else:
            runnable.append(entry)
    return runnable, already, skipped


def _estimate(entry: Planned, store: Store) -> float | None:
    """Tokens to produce, over the rate this model last showed. Falling back to the shape's
    own ceiling when there is no history, and to nothing at all when there is no ceiling —
    a bar drawn from an invented total is worse than no bar."""
    if entry.shape is None or not isinstance(entry.body, SpeedResult):
        return None
    tokens = entry.shape.rounds * entry.shape.generate * entry.shape.concurrency
    history = [
        run.result.decode_tps
        for run in store.runs_of("speed", entry.model)
        if isinstance(run.result, SpeedResult) and run.result.decode_tps
    ]
    rate = history[0] if history else entry.body.ceiling_tps
    return None if not rate else tokens / rate


def _budget(store: Store) -> int:
    """What the daemon will actually let a model occupy, so the sheet and the loader refuse
    the same shapes."""
    return current(store).memory_limit_bytes


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


@router.post("/admin/benchmarks/plan")
def plan(body: BenchmarkRequest, store: StoreDep) -> Plan:
    """The same expansion the job runs, priced and with nothing written."""
    planned = expand(body, catalog.scan(), _budget(store))
    runnable, already, skipped = _split(planned, store, body.skip_if_measured)
    estimates = [_estimate(entry, store) for entry in runnable]
    known = [value for value in estimates if value is not None]
    return Plan(
        shapes=[PlannedShape(model=entry.model, key=entry.key) for entry in runnable],
        skipped=[
            SkippedShape(
                model=entry.model,
                key=entry.key,
                reason=entry.refusal.reason,
                detail=entry.refusal.detail,
                needed_bytes=entry.refusal.needed_bytes,
                budget_bytes=entry.refusal.budget_bytes,
            )
            for entry in skipped
            if entry.refusal is not None
        ],
        already=[PlannedShape(model=entry.model, key=entry.key) for entry in already],
        warnings=[
            SkippedShape(
                model=entry.model, key=entry.key, reason="shape_mismatch", detail=entry.warning
            )
            for entry in planned
            if entry.warning is not None
        ],
        estimate_seconds=sum(known) if known else None,
    )


def _residents(engine: Engine) -> str:
    """Who else was holding memory while this was measured. A number taken with 60 GB of
    somebody else's weights resident is not the same number."""
    return json.dumps(sorted(engine.residency))


def _record(
    store: Store,
    engine: Engine,
    entry: Planned,
    state: RunState,
    reason: str | None,
    body: Result,
    temperature: float | None,
) -> None:
    header = Run(
        id=uuid.uuid4().hex,
        kind=entry.kind,
        model=entry.model,
        key=entry.key,
        state=state,
        reason=reason,
        engine_version=version("mlx_omnia"),
        mlx_version=mx.__version__,
        temp_c_start=temperature,
        residents=_residents(engine),
        created_at=time.time(),
    )
    store.insert_run(header, body)


def _work(
    engine: Engine,
    store: Store,
    body: SpeedRequest | FidelityRequest | QualityRequest,
    planned: Sequence[Planned],
    skip_if_measured: bool,
) -> Work:
    def work(job: Job) -> None:
        runnable, _already, skipped = _split(planned, store, skip_if_measured)
        total = len(runnable) + len(skipped)
        job.report(Progress(message="reserving the queue", total=float(total)))
        for entry in skipped:
            assert entry.refusal is not None
            _record(store, engine, entry, "not_run", entry.refusal.reason, entry.body, None)
        # The reservation is taken here and given back in the `finally` of the context
        # manager, whatever ends the block — a benchmark that raised must not leave the
        # daemon deaf to every other request.
        token = asyncio.run_coroutine_threadsafe(engine.acquire_queue(), job.loop).result()
        # Once for the batch, not once per shape: the walk stats a few hundred files and
        # reads every header, and the disk it asks for is the disk the measurement under it
        # is also asking for. Nothing can add a checkpoint while the queue is held.
        priced = _facts(catalog.scan())
        # One warm-up per checkpoint, not one per shape: what it buys is compiled into the
        # process and outlives the unload each shape does, so paying it again would only be a
        # round the batch spends to throw away.
        warmed: set[str] = set()
        try:
            for index, entry in enumerate(runnable):
                frames = speed.Report(
                    prefix=f"{entry.model} · {entry.key}",
                    completed=float(len(skipped) + index),
                    total=float(total),
                )
                job.report(frames.frame("starting"))
                assert entry.shape is not None, "only a speed shape reaches the runner"
                facts = priced[entry.model]
                try:
                    taken = speed.measure(
                        job,
                        engine,
                        entry.model,
                        entry.shape,
                        facts,
                        _budget(store),
                        token,
                        frames,
                        entry.model not in warmed,
                    )
                except Cancelled:
                    raise
                except Exception as error:
                    # One checkpoint that will not load must not take the batch with it: the
                    # shape gets an `error` row naming what happened, and the rest of the
                    # product still runs. Losing four measured shapes to the fifth's
                    # tokenizer is the failure mode this catch exists for.
                    _record(
                        store,
                        engine,
                        entry,
                        "error",
                        f"{type(error).__name__}: {error}",
                        entry.body,
                        None,
                    )
                    continue
                if taken.state == "ok":
                    warmed.add(entry.model)
                _record(
                    store,
                    engine,
                    entry,
                    taken.state,
                    taken.reason,
                    taken.result,
                    taken.temperature,
                )
        finally:
            asyncio.run_coroutine_threadsafe(engine.release_queue(token), job.loop).result()
        # TODO(59.8): score the quality rows here — for each planned dataset, read the
        # split through `datasets.parquet_path`, render every item with its template and
        # take the log-likelihood of each continuation under the model, then rewrite the
        # row with `accuracy`, `correct` and the breakdown. Until then the rows above stand
        # as `not_run` and the job says so.
        if isinstance(body, QualityRequest):
            raise NotImplementedError(QUALITY_TODO)
        # TODO(59.9): measure the fidelity rows here — build or reuse the reference's cached
        # top-k logits through `fidelity.cache_path`, run the candidate teacher-forced over
        # the same corpus, and rewrite the row with KL mean and p95, top-1, top-5 and Δppl.
        if isinstance(body, FidelityRequest):
            raise NotImplementedError(FIDELITY_TODO)

    return work


@router.post("/admin/benchmarks", status_code=202)
async def create(
    body: BenchmarkRequest, engine: EngineDep, store: StoreDep, registry: JobsDep
) -> JSONResponse:
    """On the loop: `start` captures it, and the work drives the engine's queue back through
    it from the thread it runs in."""
    planned = expand(body, catalog.scan(), _budget(store))
    models = sorted({entry.model for entry in planned})
    work = _work(engine, store, body, planned, body.skip_if_measured)
    return accepted(registry.start(BenchmarkJob(kind=body.kind, models=models), work))


class RunView(BaseModel):
    """The header and the body of one measurement in one object — what the views already
    hand back, including the `not_run` rows with their reason."""

    id: str
    kind: BenchmarkKind
    model: str
    key: str
    state: str
    reason: str | None
    engine_version: str
    mlx_version: str
    temp_c_start: float | None
    residents: str
    created_at: float
    speed: SpeedResult | None = None
    quality: QualityResult | None = None
    fidelity: FidelityResult | None = None


def _view(entry: BenchmarkRun) -> RunView:
    run = entry.run
    return RunView(
        id=run.id,
        kind=run.kind,
        model=run.model,
        key=run.key,
        state=run.state,
        reason=run.reason,
        engine_version=run.engine_version,
        mlx_version=run.mlx_version,
        temp_c_start=run.temp_c_start,
        residents=run.residents,
        created_at=run.created_at,
        speed=entry.result if isinstance(entry.result, SpeedResult) else None,
        quality=entry.result if isinstance(entry.result, QualityResult) else None,
        fidelity=entry.result if isinstance(entry.result, FidelityResult) else None,
    )


def _read(store: Store, kind: BenchmarkKind, key: str | None, model: str | None) -> list[RunView]:
    if key is not None:
        found = [
            entry for entry in store.runs_by_key(kind, key) if model in (None, entry.run.model)
        ]
    elif model is not None:
        found = store.runs_of(kind, model)
    else:
        found = store.runs(kind)
    return [_view(entry) for entry in found]


@router.get("/admin/benchmarks/runs")
def runs(
    store: StoreDep,
    kind: BenchmarkKind = "speed",
    key: str | None = None,
    model: str | None = None,
) -> list[RunView]:
    """`key` is what the comparison band asks with and answers one row per model, the most
    recent. `model` without a key is the other direction: one checkpoint over time."""
    return _read(store, kind, key, model)


_CSV_COLUMNS = (
    "id",
    "kind",
    "model",
    "key",
    "state",
    "reason",
    "engine_version",
    "mlx_version",
    "temp_c_start",
    "created_at",
    "context",
    "generate",
    "concurrency",
    "load_s",
    "prefill_tps",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "decode_tps",
    "step_weight_bytes",
    "step_kv_bytes",
    "ceiling_tps",
    "ceiling_fraction",
    "dataset",
    "items",
    "accuracy",
    "reference",
    "corpus",
    "kl_mean",
    "top1",
    "ppl_delta",
)


@router.get("/admin/benchmarks/runs.csv")
def runs_csv(
    store: StoreDep,
    kind: BenchmarkKind = "speed",
    key: str | None = None,
    model: str | None = None,
) -> StreamingResponse:
    """The Export button. Flat columns and no detail blobs: a spreadsheet has no use for the
    per-round JSON, and a cell holding it makes the file unreadable in the reader it was
    exported for."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)
    for view in _read(store, kind, key, model):
        flat = {
            **view.model_dump(exclude={"speed", "quality", "fidelity", "residents"}),
            **(view.speed.__dict__ if view.speed else {}),
            **(view.quality.__dict__ if view.quality else {}),
            **(view.fidelity.__dict__ if view.fidelity else {}),
        }
        writer.writerow([flat.get(column) for column in _CSV_COLUMNS])
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"content-disposition": 'attachment; filename="benchmarks.csv"'},
    )


@router.get("/admin/benchmarks/runs/{run_id}")
def one(run_id: str, store: StoreDep) -> RunView:
    entry = store.run(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no benchmark run {run_id!r}")
    return _view(entry)


@router.delete("/admin/benchmarks/runs/{run_id}", status_code=204)
def remove(run_id: str, store: StoreDep) -> None:
    """The body goes with the header, through the cascade the store's own pragma turns on."""
    if not store.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"no benchmark run {run_id!r}")
