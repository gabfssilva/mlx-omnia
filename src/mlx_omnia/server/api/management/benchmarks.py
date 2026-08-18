"""What a benchmark is asked for, what the plan says it would cost, and the job that runs it."""

from __future__ import annotations

import asyncio
import threading
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.api.management.common import EngineDep, JobsDep, accepted, budget
from mlx_omnia.server.services import benchmarks, catalog
from mlx_omnia.server.services import jobs as jobs_service
from mlx_omnia.server.services import speed as speed_service

router = APIRouter()


class _Task:
    """A job as the benchmark runner's `Task`: the two services keep their own `Progress`, and
    this is the one place the two spellings meet."""

    def __init__(self, job: jobs_service.Job) -> None:
        self._job = job

    @property
    def id(self) -> str:
        return self._job.id

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._job.loop

    @property
    def cancelled(self) -> threading.Event:
        return self._job.cancelled

    def report(self, progress: speed_service.Progress) -> None:
        self._job.report(
            jobs_service.Progress(
                message=progress.message, completed=progress.completed, total=progress.total
            )
        )


class SamplingBody(BaseModel):
    """The sampler the rounds decode under. The defaults are the deterministic end of every
    dialect: a benchmark that says nothing about sampling is measuring an argmax."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = None

    def spec(self) -> benchmarks.Sampling:
        return benchmarks.Sampling(
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


def _spec(body: SpeedRequest | FidelityRequest | QualityRequest) -> benchmarks.Spec:
    match body:
        case SpeedRequest():
            return benchmarks.SpeedSpec(
                models=body.models,
                contexts=body.contexts,
                generates=body.generates,
                concurrencies=body.concurrencies,
                rounds=body.rounds,
                sampling=body.sampling.spec(),
                page_cache=body.page_cache,
                thermal_gate_c=body.thermal_gate_c,
                stream_source=body.stream_source,
                skip_if_measured=body.skip_if_measured,
            )
        case FidelityRequest():
            return benchmarks.FidelitySpec(
                pairs=[
                    benchmarks.Pair(model=pair.model, reference=pair.reference)
                    for pair in body.pairs
                ],
                corpus=body.corpus,
                tokens=body.tokens,
                seed=body.seed,
                topk=body.topk,
                skip_if_measured=body.skip_if_measured,
            )
        case QualityRequest():
            return benchmarks.QualitySpec(
                models=body.models,
                datasets=body.datasets,
                items=body.items,
                seed=body.seed,
                shots=body.shots,
                scoring=body.scoring,
                skip_if_measured=body.skip_if_measured,
            )


async def _expanded(
    body: SpeedRequest | FidelityRequest | QualityRequest,
) -> tuple[benchmarks.Spec, list[benchmarks.Planned]]:
    spec = _spec(body)
    try:
        return spec, benchmarks.expand(spec, catalog.scan(), await budget())
    except benchmarks.Invalid as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except benchmarks.Unknown as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


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
    """What the sheet draws before anybody presses run. The three reasons for not running are
    separate because the sheet separates them: too big, already answered, and not comparable."""

    shapes: list[PlannedShape]
    skipped: list[SkippedShape]
    already: list[PlannedShape]
    warnings: list[SkippedShape]
    estimate_seconds: float | None


@router.post("/admin/benchmarks/plan")
async def plan(body: BenchmarkRequest) -> Plan:
    """The same expansion the job runs, priced and with nothing written."""
    _spec_of, planned = await _expanded(body)
    parts = await benchmarks.split(planned, body.skip_if_measured)
    estimates = [await benchmarks.estimate(entry) for entry in parts.runnable]
    known = [value for value in estimates if value is not None]
    return Plan(
        shapes=[PlannedShape(model=entry.model, key=entry.key) for entry in parts.runnable],
        skipped=[
            SkippedShape(
                model=entry.model,
                key=entry.key,
                reason=entry.refusal.reason,
                detail=entry.refusal.detail,
                needed_bytes=entry.refusal.needed_bytes,
                budget_bytes=entry.refusal.budget_bytes,
            )
            for entry in parts.skipped
            if entry.refusal is not None
        ],
        already=[PlannedShape(model=entry.model, key=entry.key) for entry in parts.already],
        warnings=[
            SkippedShape(
                model=entry.model, key=entry.key, reason="shape_mismatch", detail=entry.warning
            )
            for entry in planned
            if entry.warning is not None
        ],
        estimate_seconds=sum(known) if known else None,
    )


@router.post("/admin/benchmarks", status_code=202)
async def benchmark(body: BenchmarkRequest, engine: EngineDep, registry: JobsDep) -> JSONResponse:
    """On the loop: `start` captures it, and the work drives the engine's queue back through it
    from the thread it runs in."""
    spec, planned = await _expanded(body)
    models = sorted({entry.model for entry in planned})
    work = benchmarks.work(
        engine, spec, planned, catalog.scan, await budget(), body.skip_if_measured
    )

    def body_of(job: jobs_service.Job) -> None:
        work(_Task(job))

    job = registry.start(jobs_service.Benchmark(kind=body.kind, models=models), body_of)
    return accepted(job)
