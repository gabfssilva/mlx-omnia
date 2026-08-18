from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import TypedDict

from mlx_omnia.server.db.models.jobs import Job as JobRow
from mlx_omnia.server.db.models.jobs import JobState

TERMINAL: frozenset[JobState] = frozenset({"ok", "error", "cancelled"})

_ABANDONED = "the daemon stopped before this job finished"


@dataclass(frozen=True)
class Progress:
    """`total` is absent while it is unknown — a bar drawn from a guessed total is worse
    than no bar."""

    message: str = ""
    completed: float = 0.0
    total: float | None = None


@dataclass(frozen=True)
class Download:
    model: str


@dataclass(frozen=True)
class Load:
    model: str


@dataclass(frozen=True)
class Quantize:
    model: str
    target: str


@dataclass(frozen=True)
class Bench:
    model: str


@dataclass(frozen=True)
class Benchmark:
    """A batch, which is the one subject that is not a single checkpoint."""

    kind: str
    models: list[str]


Subject = Download | Load | Quantize | Bench | Benchmark
"""What the job is about, so a screen can draw a job beside the thing it acts on."""


def kind_of(subject: Subject) -> str:
    match subject:
        case Download():
            return "download"
        case Load():
            return "load"
        case Quantize():
            return "quantize"
        case Bench():
            return "bench"
        case Benchmark():
            return "benchmark"


def _subject(kind: str, text: str) -> Subject:
    match (kind, json.loads(text)):
        case ("download", {"model": str(model)}):
            return Download(model=model)
        case ("load", {"model": str(model)}):
            return Load(model=model)
        case ("quantize", {"model": str(model), "target": str(target)}):
            return Quantize(model=model, target=target)
        case ("bench", {"model": str(model)}):
            return Bench(model=model)
        case ("benchmark", {"kind": str(kind), "models": list(models)}) if all(
            isinstance(name, str) for name in models
        ):
            return Benchmark(kind=kind, models=[name for name in models if isinstance(name, str)])
        case other:
            raise ValueError(f"malformed job subject: {other!r}")


def _progress(text: str) -> Progress:
    match json.loads(text):
        case {
            "message": str(message),
            "completed": (int() | float()) as completed,
            "total": (None | int() | float()) as total,
        }:
            return Progress(message=message, completed=completed, total=total)
        case other:
            raise ValueError(f"malformed job progress: {other!r}")


def _state(text: str) -> JobState:
    match text:
        case "pending" | "running" | "ok" | "error" | "cancelled":
            return text
        case other:
            raise ValueError(f"malformed job state: {other!r}")


@dataclass(frozen=True)
class JobView:
    """One shape for the four windows on a job: the `202`, the `GET`, the listing and
    every frame of the stream."""

    id: str
    kind: str
    subject: Subject
    state: JobState
    progress: Progress
    created_at: float
    updated_at: float
    error: str | None = None


def _view(row: JobRow) -> JobView:
    return JobView(
        id=row.id,
        kind=row.kind,
        subject=_subject(row.kind, row.subject),
        state=_state(row.state),
        progress=_progress(row.progress),
        created_at=row.created_at,
        updated_at=row.updated_at,
        error=row.error,
    )


class _Columns(TypedDict):
    kind: str
    subject: str
    state: str
    progress: str
    created_at: float
    updated_at: float
    error: str | None


def _columns(view: JobView) -> _Columns:
    return {
        "kind": view.kind,
        "subject": json.dumps(asdict(view.subject)),
        "state": view.state,
        "progress": json.dumps(asdict(view.progress)),
        "created_at": view.created_at,
        "updated_at": view.updated_at,
        "error": view.error,
    }


async def write(view: JobView) -> None:
    columns = _columns(view)
    if not await JobRow.objects.filter(id=view.id).update(**columns):
        await JobRow(id=view.id, **columns).save()


async def abandon() -> int:
    """Every job left mid-flight by a process that is gone, marked as what it is. A row
    reaches this state only through a kill the daemon could not answer, and nothing else
    ever reconciles it."""
    return await JobRow.objects.filter(state__in=["pending", "running"]).update(
        each=False, state="error", error=_ABANDONED, updated_at=time.time()
    )


async def view(job_id: str) -> JobView | None:
    row = await JobRow.objects.get_or_none(id=job_id)
    return None if row is None else _view(row)


async def recent(active: bool = False) -> list[JobView]:
    """Newest first. `active` is the app's progress list; the unfiltered one is the
    history, which is what a client that missed a job reads."""
    rows = await JobRow.objects.order_by(["-created_at", "-id"]).all()
    views = [_view(row) for row in rows]
    return [found for found in views if not active or found.state not in TERMINAL]
