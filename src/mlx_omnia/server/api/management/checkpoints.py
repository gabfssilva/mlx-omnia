"""Tokenization, and what one checkpoint answers about itself — down to letting go of it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from mlx_omnia.engine.checkpoint import ImageCost
from mlx_omnia.engine.graph import Graph
from mlx_omnia.engine.language import tokenizer_of
from mlx_omnia.server.api.management.common import EngineDep
from mlx_omnia.server.services import catalog, prefixes

router = APIRouter()


class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


@dataclass(frozen=True)
class Tokens:
    ids: list[int]


@router.post("/admin/models/{model_id:path}/tokenize")
async def tokenize(model_id: str, body: TokenizeRequest, engine: EngineDep) -> Tokens:
    """`resolve` is reached only for a model the check above found resident, and for one of
    those it returns before it can suspend — so this route cannot become the load it refused.

    The encode goes to a thread: the BPE is Python, a full context is hundreds of thousands of
    characters, and the loop it would otherwise sit on is carrying a token stream.
    """
    if model_id not in engine.resident:
        raise HTTPException(
            status_code=409,
            detail=f"{model_id!r} is not resident: load it before tokenizing with it",
        )
    tokenizer = tokenizer_of(await engine.resolve(model_id))
    if tokenizer is None:
        raise HTTPException(status_code=500, detail=f"{model_id!r} exposes no tokenizer")

    # Read out in the thread and not after it: the tokenizer hands the ids over as it makes
    # them, so `to_thread` on `encode` alone would move a generator and leave the work here.
    def encode() -> list[int]:
        return list(tokenizer.encode(body.text))

    return Tokens(ids=await asyncio.to_thread(encode))


def _unknown(model_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{model_id!r} is not in the catalog")


@router.get("/admin/models/{model_id:path}/image")
def image(model_id: str, height: int, width: int) -> ImageCost:
    """What one image of this size would cost this checkpoint, before it is sent. The
    arithmetic stays here because it is the family's and not the dialect's."""
    try:
        return catalog.image_cost(model_id, height, width)
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error
    except catalog.ImageSizeInvalid as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except catalog.TakesNoImage as error:
        raise HTTPException(status_code=409, detail=f"{model_id!r} takes no image") from error


@router.get("/admin/models/{model_id:path}/card", response_class=PlainTextResponse)
def card(model_id: str) -> str:
    """The checkpoint's README, raw — rendering it is the client's job."""
    try:
        return catalog.card(model_id)
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error
    except catalog.NoModelCard as error:
        raise HTTPException(status_code=404, detail=f"{model_id!r} has no model card") from error


@router.get("/admin/models/{model_id:path}/files")
def files(model_id: str) -> list[catalog.CheckpointFile]:
    try:
        return catalog.files(model_id)
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error


@router.get("/admin/models/{model_id:path}/blueprint")
def blueprint(model_id: str) -> Graph:
    """What this checkpoint's decode step is made of: the trunk, one graph per kind of block,
    and the kernel each declared operation resolved to. Nothing is loaded to answer it."""
    try:
        return catalog.blueprint(model_id)
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error
    except catalog.NotTraceable as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/admin/models/{model_id:path}/assets/{asset:path}")
def asset(model_id: str, asset: str) -> FileResponse:
    """A file the card references relatively (its images). The name resolves inside the
    checkpoint and nowhere else — `..` and absolute paths are refused before the disk is
    asked."""
    try:
        return FileResponse(catalog.asset(model_id, asset))
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error
    except catalog.NoSuchAsset as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/admin/models/{model_id:path}")
async def model(model_id: str, engine: EngineDep) -> catalog.CatalogEntry:
    try:
        return catalog.model(model_id, catalog.resident_bytes(engine))
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error


@router.delete("/admin/models/{model_id:path}", status_code=204)
async def remove_model(model_id: str, engine: EngineDep) -> None:
    """The weights and everything keyed to them — the conversations spilled under this id
    included."""
    try:
        await catalog.remove(model_id, catalog.resident_bytes(engine), prefixes.forget)
    except catalog.UnknownModel as error:
        raise _unknown(model_id) from error
    except catalog.ModelResident as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
