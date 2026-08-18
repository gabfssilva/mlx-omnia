"""The dependencies every `/admin` route resolves through, and the two answers they share."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from mlx_omnia.server.events import Hub
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services import config as settings
from mlx_omnia.server.services import jobs as jobs_service

STARTED = time.monotonic()

KEEP_ALIVE_SECONDS = 15.0

type BenchmarkKind = Literal["speed", "quality", "fidelity"]
type Tier = Literal["memory", "disk"]


async def _engine_of(request: Request) -> Engine:
    """Async so it reads the engine's dicts on the loop that mutates them."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


async def _jobs_of(request: Request) -> jobs_service.Jobs:
    registry = request.app.state.jobs
    assert isinstance(registry, jobs_service.Jobs)
    return registry


async def _hub_of(request: Request) -> Hub:
    hub = request.app.state.events
    assert isinstance(hub, Hub)
    return hub


async def _metrics_of(request: Request) -> Metrics:
    register = request.app.state.metrics
    assert isinstance(register, Metrics)
    return register


def _host_of(request: Request) -> str:
    host = request.app.state.host
    assert isinstance(host, str)
    return host


EngineDep = Annotated[Engine, Depends(_engine_of)]
JobsDep = Annotated[jobs_service.Jobs, Depends(_jobs_of)]
HubDep = Annotated[Hub, Depends(_hub_of)]
MetricsDep = Annotated[Metrics, Depends(_metrics_of)]
HostDep = Annotated[str, Depends(_host_of)]


async def budget() -> int:
    """What the daemon will let a model occupy, so the sheet and the loader refuse the same
    shapes."""
    return (await settings.current()).memory_limit_bytes


def accepted(job: jobs_service.Job) -> JSONResponse:
    """What every creator answers. The first frame travels in the body so the client knows the
    id and the state without a second round trip to the `Location` it was just given."""
    return JSONResponse(
        status_code=202,
        content=asdict(job.view),
        headers={"Location": f"/admin/jobs/{job.id}"},
    )
