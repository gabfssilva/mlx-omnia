"""Residency by hand: the two buttons the model screen has, and what each one costs.

Loading is a job (34.1's), because a cold 30B is seconds and a request that waits for it is a
request that times out. A model that is already resident answers the same 202 over a job that
finishes immediately: `Engine.resolve` hands back what it holds, so the button is a no-op and
never a second load. A name no checkpoint answers to is not refused here either — the loader
raises inside the job, and the reason reaches the client as the job's error instead of as a
status code the PUT invented before trying.

Unloading is not a job. What it costs is waiting for the gate, and what the caller wants back
is whether the memory returned — which a 204 says and a job id does not.

Mounted **after `profiles.router` and before `catalog.router`**, which is the one order that
holds: `{model_id:path}` matches slashes at both ends of the collision. Registered after the
catalog, a `DELETE /admin/models/a/b/residency` is the catalog's own delete reading
`a/b/residency` as the model's name; registered before the profiles, this route answers
`PUT /admin/models/m/profiles/residency` with a load of a model called `m/profiles`.
"""

import asyncio
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from mlx_omnia.server.engine import Engine
from mlx_omnia.server.jobs import Job, JobsDep, Load, Progress, Work, accepted


async def _engine(request: Request) -> Engine:
    """Async so it reads the engine's dicts on the loop that mutates them."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


async def _resolve(engine: Engine, model_id: str, cancelled: threading.Event) -> None:
    """Answers nothing on purpose: the future the worker thread waits on keeps whatever the
    coroutine returned, and a second reference to a 30B living until the job's frame dies is
    the whole failure this route is supposed to avoid.

    A cancel that lands while the weights are being read gives them back. An MLX load does
    not answer a flag from outside, so the flag cannot stop it — and the job would otherwise
    end labelled `cancelled` with the model resident and the memory taken, which is the label
    saying the opposite of what the disk and `/admin/state` say. What cancelling a load means
    is "do not keep this", and that is what this does.
    """
    await engine.resolve(model_id)
    if cancelled.is_set():
        await engine.unload(model_id)


def _load(engine: Engine, model_id: str) -> Work:
    def work(job: Job) -> None:
        # A report first, as every job body begins: one cancelled while it waited for a
        # thread must not start paying for the load.
        job.report(Progress(message=f"loading {model_id}"))
        # The engine's dicts belong to the loop, and this is a worker thread. The load itself
        # still happens off the loop: `resolve` hands the blocking part to a thread of its own.
        asyncio.run_coroutine_threadsafe(
            _resolve(engine, model_id, job.cancelled), job.loop
        ).result()

    return work


@router.put("/admin/models/{model_id:path}/residency", status_code=202)
async def load(model_id: str, engine: EngineDep, registry: JobsDep) -> JSONResponse:
    return accepted(registry.start(Load(model=model_id), _load(engine, model_id)))


@router.delete("/admin/models/{model_id:path}/residency", status_code=204)
async def unload(model_id: str, engine: EngineDep) -> None:
    """Interrupts nothing and refuses nothing already asked for: the generation in flight ends
    with every token it had left, and a request already queued for this model runs as well —
    it took its own reference to it at `submit`. The 204 says the engine's last reference is
    gone; the bytes a queued request is still holding come back when that request ends."""
    if not await engine.unload(model_id):
        raise HTTPException(status_code=404, detail=f"{model_id!r} is not resident")
