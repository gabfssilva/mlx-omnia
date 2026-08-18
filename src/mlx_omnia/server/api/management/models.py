"""Bringing a checkpoint down, and the catalog listing it lands in."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from mlx_omnia.server.api.management.common import EngineDep, JobsDep, accepted
from mlx_omnia.server.services import catalog, downloads

router = APIRouter()


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    quant: str | None = None


@router.post("/admin/models", status_code=202)
async def download(body: DownloadRequest, registry: JobsDep) -> JSONResponse:
    """On the loop, because `start` captures it to hand frames back from the worker thread —
    which is also what makes the checks below a gate."""
    try:
        job = await downloads.start_download(registry, body.repo, body.quant)
    except downloads.AlreadyDownloading as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except downloads.BeingCollected as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except downloads.AlreadyOnDisk as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return accepted(job)


@router.get("/admin/models")
async def models(engine: EngineDep, resident: bool = False) -> list[catalog.CatalogEntry]:
    return catalog.models(catalog.resident_bytes(engine), only_resident=resident)
