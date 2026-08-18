"""The history the batch writes, and the reads that join a header to its body."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from importlib.metadata import version

import mlx.core as mx

from mlx_omnia.server.db.models.benchmarks import (
    BenchmarkKind,
    BenchmarkRun,
    FidelityResult,
    QualityResult,
    RunState,
    SpeedResult,
)
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services.benchmarks.specs import Body, Measured, Planned


def _residents(engine: Engine) -> str:
    """Who else was holding memory while this was measured. A number taken with 60 GB of
    somebody else's weights resident is not the same number."""
    return json.dumps(sorted(engine.residency))


def _kind_of(result: Body) -> BenchmarkKind:
    match result:
        case SpeedResult():
            return "speed"
        case QualityResult():
            return "quality"
        case FidelityResult():
            return "fidelity"


async def insert_run(run: BenchmarkRun, result: Body) -> None:
    """The only door into the benchmark tables, and it takes both halves: two calls would
    eventually be used as one — a header written, a body forgotten — and the result is a run no
    view can see. The kind is taken from the result's own type, so a `speed` header carrying a
    quality body cannot be written at all."""
    kind = _kind_of(result)
    if kind != run.kind:
        raise ValueError(f"{run.kind!r} header carries a {kind!r} result")
    await run.save()
    result.run_id = run.id
    await result.save()


async def record(
    engine: Engine,
    entry: Planned,
    state: RunState,
    reason: str | None,
    body: Body,
    temperature: float | None,
) -> None:
    header = BenchmarkRun(
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
    await insert_run(header, body)


async def _bodies(kind: BenchmarkKind, ids: Sequence[str]) -> Mapping[str, Body]:
    if not ids:
        return {}
    found: Sequence[Body]
    match kind:
        case "speed":
            found = await SpeedResult.objects.filter(run_id__in=list(ids)).all()
        case "quality":
            found = await QualityResult.objects.filter(run_id__in=list(ids)).all()
        case "fidelity":
            found = await FidelityResult.objects.filter(run_id__in=list(ids)).all()
    return {body.run_id: body for body in found}


async def _joined(kind: BenchmarkKind, headers: Sequence[BenchmarkRun]) -> list[Measured]:
    bodies = await _bodies(kind, [header.id for header in headers])
    return [
        Measured(run=header, result=bodies[header.id]) for header in headers if header.id in bodies
    ]


async def runs_by_key(kind: BenchmarkKind, key: str) -> list[Measured]:
    """One row per model, the most recent — what the comparison band asks: *who has been
    measured under this shape*. The history of a single model is `runs_of`."""
    headers: Sequence[BenchmarkRun] = (
        await BenchmarkRun.objects.filter(kind=kind, key=key).order_by(["-created_at", "-id"]).all()
    )
    latest: dict[str, BenchmarkRun] = {}
    for header in headers:
        latest.setdefault(header.model, header)
    ordered = [latest[model] for model in sorted(latest)]
    return await _joined(kind, ordered)


async def runs_of(kind: BenchmarkKind, model: str) -> list[Measured]:
    """Every measurement of one checkpoint, newest first: the series that answers whether an
    engine version moved the number."""
    headers: Sequence[BenchmarkRun] = (
        await BenchmarkRun.objects.filter(kind=kind, model=model)
        .order_by(["-created_at", "-id"])
        .all()
    )
    return await _joined(kind, headers)


async def runs(kind: BenchmarkKind) -> list[Measured]:
    headers: Sequence[BenchmarkRun] = (
        await BenchmarkRun.objects.filter(kind=kind).order_by(["-created_at", "-id"]).all()
    )
    return await _joined(kind, headers)


async def run(run_id: str) -> Measured | None:
    """By id, without the caller having to know the kind: the header answers that."""
    header = await BenchmarkRun.objects.get_or_none(id=run_id)
    if header is None:
        return None
    bodies = await _bodies(_kind(header.kind), [run_id])
    body = bodies.get(run_id)
    return None if body is None else Measured(run=header, result=body)


def _kind(value: str) -> BenchmarkKind:
    match value:
        case "speed" | "quality" | "fidelity":
            return value
        case _:
            raise ValueError(f"{value!r} is not a benchmark kind")


async def measured(kind: BenchmarkKind, model: str, key: str) -> bool:
    """What `skip_if_measured` asks. `not_run` counts as measured: the shape was decided
    against once, and deciding again costs the same arithmetic for the same answer."""
    return (
        await BenchmarkRun.objects.filter(kind=kind, model=model, key=key)
        .exclude(state="error")
        .exists()
    )


async def delete_run(run_id: str) -> bool:
    """The body goes with the header, through the cascade the file carries."""
    return await BenchmarkRun.objects.delete(id=run_id) == 1
