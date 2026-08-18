"""Everything named under a model id: its profiles, its residency, its switches, its sampler.

Under the profile routes and above the catalog's, which is the one order that holds:
`{model_id:path}` matches slashes at both ends of the collision.
"""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.api.management.common import EngineDep, JobsDep, accepted
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services import features as features_service
from mlx_omnia.server.services import jobs as jobs_service
from mlx_omnia.server.services import profiles as profiles_service
from mlx_omnia.server.services.features import Features

router = APIRouter()


class ProfileBody(BaseModel):
    """Unknown fields are refused rather than dropped, `template` among them: a profile that
    accepted one and never rendered it would be a client told, wrongly, that its template is in
    use."""

    model_config = ConfigDict(extra="forbid")

    sampling: profiles_service.Sampling = profiles_service.Sampling()
    system_prompt: str | None = None
    features: Features = Features()


@router.get("/admin/models/{model_id:path}/profiles/{name}")
async def profile(model_id: str, name: str) -> profiles_service.ProfileView:
    try:
        return await profiles_service.profile(model_id, name)
    except profiles_service.NoSuchProfile as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.put("/admin/models/{model_id:path}/profiles/{name}")
async def save_profile(model_id: str, name: str, body: ProfileBody) -> profiles_service.ProfileView:
    """The model is not checked against the catalog: a profile written before its download
    lands is not an error, and a dialect lists a profile only under a model it can see."""
    try:
        return await profiles_service.save(
            model_id, name, body.sampling, body.system_prompt, body.features
        )
    except profiles_service.InvalidProfileName as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except profiles_service.SpeculationLocked as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.delete("/admin/models/{model_id:path}/profiles/{name}", status_code=204)
async def remove_profile(model_id: str, name: str) -> None:
    try:
        await profiles_service.remove(model_id, name)
    except profiles_service.NoSuchProfile as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


async def _resolve(engine: Engine, model_id: str, cancelled: threading.Event) -> None:
    """Answers nothing on purpose: the future the worker thread waits on keeps whatever the
    coroutine returned, and a second reference to a 30B living until the job's frame dies is
    the failure this route exists to avoid.

    A cancel that lands while the weights are being read gives them back: an MLX load does not
    answer a flag from outside, and the job would otherwise end labelled `cancelled` with the
    model resident.
    """
    await engine.resolve(model_id)
    if cancelled.is_set():
        await engine.unload(model_id)


def _load(engine: Engine, model_id: str) -> jobs_service.Work:
    def work(job: jobs_service.Job) -> None:
        job.report(jobs_service.Progress(message=f"loading {model_id}"))
        # The engine's dicts belong to the loop, and this is a worker thread. The load itself
        # still happens off the loop: `resolve` hands the blocking part to a thread of its own.
        asyncio.run_coroutine_threadsafe(
            _resolve(engine, model_id, job.cancelled), job.loop
        ).result()

    return work


@router.put("/admin/models/{model_id:path}/residency", status_code=202)
async def load(model_id: str, engine: EngineDep, registry: JobsDep) -> JSONResponse:
    return accepted(registry.start(jobs_service.Load(model=model_id), _load(engine, model_id)))


@router.delete("/admin/models/{model_id:path}/residency", status_code=204)
async def unload(model_id: str, engine: EngineDep) -> None:
    """Interrupts nothing and refuses nothing already asked for: the generation in flight ends
    with every token it had left, and a request already queued for this model runs as well."""
    if not await engine.unload(model_id):
        raise HTTPException(status_code=404, detail=f"{model_id!r} is not resident")


class SettingsView(BaseModel):
    """One shape for the `GET` and the `PUT`. `available` is derived from the catalog on
    every read — a drafter deleted from disk turns the switch unavailable without anything
    having to rewrite the row that named it."""

    model: str
    features: Features
    max_concurrent_requests: int | None = None
    available: list[str]
    mtp_available: bool = False
    unavailable_reason: str | None = None


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Features = Features()
    max_concurrent_requests: int | None = Field(default=None, ge=1)


def _settings_view(
    model_id: str, switched: Features, max_concurrent_requests: int | None
) -> SettingsView:
    try:
        found = features_service.availability(model_id)
    except catalog.UnknownModel as error:
        raise HTTPException(status_code=404, detail=f"no model {model_id!r}") from error
    return SettingsView(
        model=model_id,
        features=switched,
        max_concurrent_requests=max_concurrent_requests,
        available=found.drafters,
        mtp_available=found.mtp,
        unavailable_reason=found.reason,
    )


@router.get("/admin/models/{model_id:path}/settings")
async def model_settings(model_id: str) -> SettingsView:
    saved = await features_service.settings_row(model_id)
    return _settings_view(
        model_id, features_service.parse(saved.features), saved.max_concurrent_requests
    )


@router.put("/admin/models/{model_id:path}/settings")
async def save_model_settings(model_id: str, body: SettingsBody, engine: EngineDep) -> SettingsView:
    """Stores the switches, and lets go of the model if it is holding the old ones.

    The pairing happens at load — that is where a second checkpoint enters memory — so a
    resident model goes on decoding under the settings it was loaded with. Unloading here is
    what makes the switch mean something now instead of after the next restart."""
    previous = (await features_service.settings_row(model_id)).features
    try:
        await features_service.save(model_id, body.features, body.max_concurrent_requests)
    except features_service.SettingsRefused as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    view = _settings_view(model_id, body.features, body.max_concurrent_requests)
    if body.features.model_dump_json(exclude_none=True) != previous:
        await engine.unload(model_id)
    return view


@router.get("/admin/models/{model_id:path}/sampling")
async def model_sampling(model_id: str) -> profiles_service.Sampling:
    """The model's own knobs, which every request for it gets before any profile opines."""
    return await profiles_service.defaults(model_id)


@router.put("/admin/models/{model_id:path}/sampling")
async def save_model_sampling(
    model_id: str, body: profiles_service.Sampling
) -> profiles_service.Sampling:
    """Replaces rather than patches, like a profile's `PUT`: what the body omits is what this
    daemon no longer opines on, and clearing a knob is sending it unset."""
    return await profiles_service.save_sampling(model_id, body)
