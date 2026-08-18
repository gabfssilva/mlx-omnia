"""The job registry's surface: the list, one job, its cancel, and its stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from mlx_omnia.server.api import sse
from mlx_omnia.server.api.management.common import KEEP_ALIVE_SECONDS, JobsDep
from mlx_omnia.server.services import jobs as jobs_service

router = APIRouter()


def _job_frame(view: jobs_service.JobView) -> str:
    return f"data: {json.dumps(asdict(view))}\n\n"


async def _last_frame(view: jobs_service.JobView) -> AsyncIterator[str]:
    """A job nobody runs any more: its state is the whole stream."""
    yield _job_frame(view)


async def _job_frames(job: jobs_service.Job) -> AsyncIterator[str]:
    """Subscribed before the state is read. The client walking away cancels nothing: a job
    outlives whoever was watching it."""
    with job.watch() as queue:
        current = job.view
        yield _job_frame(current)
        if current.state in jobs_service.TERMINAL:
            return
        while True:
            try:
                view = await asyncio.wait_for(queue.get(), KEEP_ALIVE_SECONDS)
            except TimeoutError:
                yield sse.KEEP_ALIVE
                continue
            if view is None:
                return
            yield _job_frame(view)


@router.get("/admin/jobs")
async def jobs(active: bool = False) -> list[jobs_service.JobView]:
    """Newest first. `active=true` is the app's progress list; the unfiltered one is the
    history, which is what a client that missed a job reads."""
    return await jobs_service.recent(active)


@router.get("/admin/jobs/{job_id}")
async def job(job_id: str, registry: JobsDep) -> jobs_service.JobView:
    try:
        return await registry.found(job_id)
    except jobs_service.NoSuchJob as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.delete("/admin/jobs/{job_id}", status_code=202)
async def cancel_job(job_id: str, registry: JobsDep) -> jobs_service.JobView:
    """202 and the job as it still is: the flag is set here, and the work turns it into
    `cancelled` when it reaches its next report."""
    try:
        return await registry.cancel(job_id)
    except jobs_service.NoSuchJob as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except jobs_service.JobFinished as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/admin/jobs/{job_id}/events")
async def job_events(job_id: str, registry: JobsDep) -> StreamingResponse:
    running = registry.live(job_id)
    if running is None:
        try:
            stream = _last_frame(await registry.found(job_id))
        except jobs_service.NoSuchJob as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
    else:
        stream = _job_frames(running)
    return StreamingResponse(stream, media_type=sse.MEDIA_TYPE)
