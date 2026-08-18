"""Quantization: the priced plan, and the job that writes one."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.api.management.common import JobsDep, accepted
from mlx_omnia.server.services import jobs as jobs_service
from mlx_omnia.server.services import quantize

router = APIRouter()


class Override(BaseModel):
    """One group of leaves against the rest of the plan. `group_size` absent is the plan's
    own."""

    model_config = ConfigDict(extra="forbid")

    bits: int
    group_size: int | None = None


class PlanRequest(BaseModel):
    """The selection alone — everything but where the result lands, which pricing does not need
    because nothing is written."""

    model_config = ConfigDict(extra="forbid")

    source: str
    mode: Literal["affine", "mxfp4", "mxfp8", "nvfp4"] = "affine"
    bits: int = 4
    group_size: int = 64
    overrides: dict[str, Override | None] = Field(default_factory=dict)
    method: Literal["rtn", "awq", "gptq", "oq", "oqe"] = "rtn"
    sequences: int = Field(default=16, gt=0)
    sequence_length: int = Field(default=256, gt=0)
    target_bpw: float | None = None
    hard_cap_bpw: float | None = None


class QuantizeRequest(PlanRequest):
    repo: str = Field(pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    """Where it lands, and the id it is served under. Two segments, and nothing that could name
    a directory outside the hub cache."""


def _selection(body: PlanRequest) -> quantize.Selection:
    """`provided` carries what the caller actually named: the refusals distinguish a default
    from a caller who believes a calibration pass will run."""
    return quantize.Selection(
        source=body.source,
        mode=body.mode,
        bits=body.bits,
        group_size=body.group_size,
        overrides={
            pattern: None
            if override is None
            else quantize.Override(bits=override.bits, group_size=override.group_size)
            for pattern, override in body.overrides.items()
        },
        method=body.method,
        sequences=body.sequences,
        sequence_length=body.sequence_length,
        target_bpw=body.target_bpw,
        hard_cap_bpw=body.hard_cap_bpw,
        provided=frozenset(body.model_fields_set),
    )


class _Reporter:
    """The service's `Reporter`, over a job's own progress door."""

    def __init__(self, job: jobs_service.Job) -> None:
        self._job = job

    def report(self, message: str, completed: float = 0.0, total: float | None = None) -> None:
        self._job.report(jobs_service.Progress(message=message, completed=completed, total=total))


@router.post("/admin/quantizations", status_code=202)
async def quantization(body: QuantizeRequest, registry: JobsDep) -> JSONResponse:
    """On the loop, which is what makes the checks below a gate: no `await` between reading
    them and registering the job that answers them."""
    selection = _selection(body)
    try:
        quantize.admissible(selection)
        formats = quantize.formats_of(selection)
    except quantize.Invalid as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        quantize.reserve(body.repo, lambda job_id: registry.live(job_id) is not None)
    except quantize.Conflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    work = quantize.work(quantize.Request(selection=selection, repo=body.repo), formats)

    def body_of(job: jobs_service.Job) -> None:
        work(_Reporter(job))

    job = registry.start(jobs_service.Quantize(model=body.source, target=body.repo), body_of)
    quantize.claim(body.repo, job.id)
    return accepted(job)


@router.post("/admin/quantizations/plan")
def price(body: PlanRequest) -> quantize.PricedPlan:
    """The same selection a job would run, resolved and costed without writing anything. Sync
    like the catalog's own handlers: building the tree of a 35B takes milliseconds, and it is
    not the loop's work."""
    try:
        return quantize.price(_selection(body))
    except quantize.Invalid as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except quantize.Unknown as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except quantize.Conflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
