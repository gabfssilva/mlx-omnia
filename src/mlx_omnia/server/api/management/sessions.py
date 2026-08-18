"""The conversations the app keeps, and the routes that hold them."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, JsonValue

from mlx_omnia.server.services import sessions as sessions_service

router = APIRouter()


class SessionBody(BaseModel):
    """The two fields a client sets. Unknown ones are refused rather than dropped."""

    model_config = ConfigDict(extra="forbid")

    title: str = "New chat"
    model: str = ""


class SessionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    model: str | None = None


class MessagesBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[JsonValue]


def _no_session(error: sessions_service.NoSuchSession) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


@router.get("/admin/sessions")
async def sessions() -> dict[str, list[sessions_service.SessionSummary]]:
    return {"sessions": await sessions_service.summaries()}


@router.post("/admin/sessions", status_code=201)
async def create_session(body: SessionBody) -> sessions_service.SessionView:
    return await sessions_service.create(body.title, body.model)


@router.get("/admin/sessions/{session_id}")
async def session(session_id: str) -> sessions_service.SessionView:
    try:
        return await sessions_service.view(session_id)
    except sessions_service.NoSuchSession as error:
        raise _no_session(error) from error


@router.patch("/admin/sessions/{session_id}")
async def update_session(session_id: str, body: SessionPatch) -> sessions_service.SessionView:
    try:
        return await sessions_service.update(session_id, body.title, body.model)
    except sessions_service.NoSuchSession as error:
        raise _no_session(error) from error


@router.put("/admin/sessions/{session_id}/messages")
async def session_messages(session_id: str, body: MessagesBody) -> sessions_service.SessionView:
    """The whole array, not an append: what the client holds is the conversation, and a turn
    edited or dropped on that side has to be able to leave the file too."""
    try:
        return await sessions_service.replace_messages(session_id, body.messages)
    except sessions_service.NoSuchSession as error:
        raise _no_session(error) from error


@router.delete("/admin/sessions/{session_id}", status_code=204)
async def remove_session(session_id: str) -> None:
    try:
        await sessions_service.remove(session_id)
    except sessions_service.NoSuchSession as error:
        raise _no_session(error) from error
