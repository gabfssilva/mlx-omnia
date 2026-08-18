"""The measurements themselves: the rows, their export, and the reference caches behind them."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_omnia.server.api.management.common import BenchmarkKind
from mlx_omnia.server.services import benchmarks

router = APIRouter()


class RunView(BaseModel):
    """The header and the body of one measurement in one object — the `not_run` rows with their
    reason included."""

    id: str
    kind: str
    model: str
    key: str
    state: str
    reason: str | None
    engine_version: str
    mlx_version: str
    temp_c_start: float | None
    residents: str
    created_at: float
    speed: dict[str, object] | None = None
    quality: dict[str, object] | None = None
    fidelity: dict[str, object] | None = None


def _run_view(entry: benchmarks.Measured) -> RunView:
    run = entry.run
    body = entry.result.model_dump(exclude={"run_id"})
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
        speed=body if run.kind == "speed" else None,
        quality=body if run.kind == "quality" else None,
        fidelity=body if run.kind == "fidelity" else None,
    )


async def _read_runs(kind: BenchmarkKind, key: str | None, model: str | None) -> list[RunView]:
    if key is not None:
        found = [
            entry
            for entry in await benchmarks.runs_by_key(kind, key)
            if model in (None, entry.run.model)
        ]
    elif model is not None:
        found = await benchmarks.runs_of(kind, model)
    else:
        found = await benchmarks.runs(kind)
    return [_run_view(entry) for entry in found]


@router.get("/admin/benchmarks/runs")
async def runs(
    kind: BenchmarkKind = "speed", key: str | None = None, model: str | None = None
) -> list[RunView]:
    """`key` is what the comparison band asks with and answers one row per model, the most
    recent. `model` without a key is the other direction: one checkpoint over time."""
    return await _read_runs(kind, key, model)


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
async def runs_csv(
    kind: BenchmarkKind = "speed", key: str | None = None, model: str | None = None
) -> StreamingResponse:
    """The Export button. Flat columns and no detail blobs: a spreadsheet has no use for the
    per-round JSON, and a cell holding it makes the file unreadable."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)
    for view in await _read_runs(kind, key, model):
        flat: dict[str, object] = {
            **view.model_dump(exclude={"speed", "quality", "fidelity", "residents"}),
            **(view.speed or {}),
            **(view.quality or {}),
            **(view.fidelity or {}),
        }
        writer.writerow([flat.get(column) for column in _CSV_COLUMNS])
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"content-disposition": 'attachment; filename="benchmarks.csv"'},
    )


@router.get("/admin/benchmarks/runs/{run_id}")
async def one_run(run_id: str) -> RunView:
    entry = await benchmarks.run(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no benchmark run {run_id!r}")
    return _run_view(entry)


@router.delete("/admin/benchmarks/runs/{run_id}", status_code=204)
async def remove_run(run_id: str) -> None:
    """The body goes with the header, through the cascade the file carries."""
    if not await benchmarks.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"no benchmark run {run_id!r}")


class ReferenceView(BaseModel):
    id: str
    reference: str
    corpus: str
    tokens: int
    seed: int
    topk: int
    bytes: int
    created_at: float
    present: bool
    """Whether the file the row points at is still there. A row without its file is not a cache
    hit, and saying so beats a listing that reports space nothing occupies."""


class References(BaseModel):
    entries: list[ReferenceView]
    total_bytes: int


@router.get("/admin/benchmarks/references")
async def references() -> References:
    entries = [
        ReferenceView(
            id=entry.id,
            reference=entry.reference,
            corpus=entry.corpus,
            tokens=entry.tokens,
            seed=entry.seed,
            topk=entry.topk,
            bytes=entry.bytes,
            created_at=entry.created_at,
            present=Path(entry.path).is_file(),
        )
        for entry in await benchmarks.references()
    ]
    return References(
        entries=entries, total_bytes=sum(entry.bytes for entry in entries if entry.present)
    )


@router.delete("/admin/benchmarks/references/{entry_id}", status_code=204)
async def remove_reference(entry_id: str) -> None:
    """The row and the file. Dropping one without the other is either a leak on disk or a hit
    that reads a path that is gone."""
    if not await benchmarks.discard_reference(entry_id):
        raise HTTPException(status_code=404, detail=f"no reference cache {entry_id!r}")
