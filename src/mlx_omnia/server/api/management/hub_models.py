"""The Hub side of downloads: what a download left staged, and what the Hub has to offer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from mlx_omnia.server.api.management.common import JobsDep
from mlx_omnia.server.services import catalog, downloads

router = APIRouter()


@router.get("/admin/hub/staging")
def staged_repositories() -> list[downloads.Staged]:
    """The bytes a download left behind, and what they cost."""
    return downloads.staged_repositories()


@router.delete("/admin/hub/staging/{repo:path}", status_code=204)
async def drop_staged(repo: str, registry: JobsDep) -> None:
    """Giving up on the resume. Refused while a job is fetching into it: `rmtree` under a
    running `hf_hub_download` is a download that dies on a directory that stopped existing."""
    try:
        await downloads.drop_staged(registry, repo)
    except downloads.AlreadyDownloading as error:
        raise HTTPException(
            status_code=409, detail=f"job {error.job_id} is downloading {repo!r}"
        ) from error
    except downloads.NothingStaged as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/admin/hub/models")
def search(query: str, limit: int = 20) -> list[downloads.HubModel]:
    """The search field's source. Sorted by downloads: the field holds a guess at a name, and
    the popular match is nearly always the one meant."""
    return downloads.search(query, limit)


@router.get("/admin/hub/models/{repo:path}/files")
def hub_files(repo: str) -> list[catalog.CheckpointFile]:
    try:
        return downloads.hub_files(repo)
    except downloads.NotOnHub as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/admin/hub/models/{repo:path}/card", response_class=PlainTextResponse)
def hub_card(repo: str) -> str:
    try:
        return downloads.hub_card(repo)
    except downloads.NoModelCard as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
