"""/admin/sessions: the conversations the app keeps between runs.

The daemon is where they live rather than the app, for the same reason the config is: two
clients on the same machine — the desktop window and a second one, or a browser — see one
history, and a reinstalled app finds its chats where it left them.

What a message *is* stays the client's: the column holds the JSON array whole, and the
only thing checked on the way in is that it is an array. The alternative is a server that
has to learn every content part a dialect grows — an image, a tool call, a reasoning block
— to be able to store a conversation it never reads. Nothing here renders these messages;
a request that wants a completion sends them to `/v1/chat/completions` itself.

`updated_at` is the sidebar's order, so every write bumps it — the rename included. What it
answers is "when did this conversation last change", and a title is part of it.
"""

import time
from dataclasses import dataclass
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from mlx_omnia.server.store import Session, SessionSummary, Store

_MESSAGES = TypeAdapter(list[JsonValue])
"""Both directions across the TEXT column. Reading is a parse and not a cast: a row somebody
hand-edited into the file fails here, named, instead of reaching a client as a string where
it expects a list."""


class SessionBody(BaseModel):
    """The two fields a client sets. Unknown ones are refused rather than dropped — a body
    carrying `messages` would otherwise be a client told, wrongly, that its conversation was
    saved, and that is what `PUT .../messages` is for."""

    model_config = ConfigDict(extra="forbid")

    title: str = "New chat"
    """The column is NOT NULL and the list is drawn from it, so a session created before
    anyone has typed still has a name."""
    model: str = ""
    """Empty until a client picks one: a conversation may outlive the checkpoint it started
    on, and the catalog is not consulted here for the same reason a profile does not consult
    it — a model that is not downloaded yet is not an error."""


class SessionPatch(BaseModel):
    """A field left out — or sent as `null`, which for these two means the same thing since
    neither has a null to mean anything by — goes on holding what the row already says."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    model: str | None = None


class MessagesBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[JsonValue]


@dataclass(frozen=True)
class SessionView:
    """The stored row with its conversation parsed: `messages` goes out as the array the
    client sent, never as the text the column holds."""

    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    messages: list[JsonValue]


def _view(session: Session) -> SessionView:
    return SessionView(
        id=session.id,
        title=session.title,
        model=session.model,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=_MESSAGES.validate_json(session.messages),
    )


def _store(request: Request) -> Store:
    store = request.app.state.store
    assert isinstance(store, Store)
    return store


StoreDep = Annotated[Store, Depends(_store)]

router = APIRouter()


def _found(store: Store, session_id: str) -> Session:
    session = store.session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return session


@router.get("/admin/sessions")
def sessions(store: StoreDep) -> dict[str, list[SessionSummary]]:
    return {"sessions": store.sessions()}


@router.post("/admin/sessions", status_code=201)
def create(body: SessionBody, store: StoreDep) -> SessionView:
    """The id is the server's — a client that generated its own could collide with a second
    one on the same file, and `INSERT OR REPLACE` would answer by overwriting a conversation
    instead of failing."""
    now = time.time()
    session = Session(
        id=uuid4().hex,
        title=body.title,
        model=body.model,
        created_at=now,
        updated_at=now,
        messages="[]",
    )
    store.save_session(session)
    return _view(session)


@router.get("/admin/sessions/{session_id}")
def session(session_id: str, store: StoreDep) -> SessionView:
    return _view(_found(store, session_id))


@router.patch("/admin/sessions/{session_id}")
def update(session_id: str, body: SessionPatch, store: StoreDep) -> SessionView:
    current = _found(store, session_id)
    renamed = Session(
        id=current.id,
        title=current.title if body.title is None else body.title,
        model=current.model if body.model is None else body.model,
        created_at=current.created_at,
        updated_at=time.time(),
        messages=current.messages,
    )
    store.save_session(renamed)
    return _view(renamed)


@router.put("/admin/sessions/{session_id}/messages")
def messages(session_id: str, body: MessagesBody, store: StoreDep) -> SessionView:
    """The whole array, not an append: what the client holds is the conversation, and a turn
    edited or dropped on that side has to be able to leave the file too. It is also what makes
    the write idempotent — a retried PUT after a dropped response says the same thing."""
    current = _found(store, session_id)
    written = Session(
        id=current.id,
        title=current.title,
        model=current.model,
        created_at=current.created_at,
        updated_at=time.time(),
        messages=_MESSAGES.dump_json(body.messages).decode(),
    )
    store.save_session(written)
    return _view(written)


@router.delete("/admin/sessions/{session_id}", status_code=204)
def remove(session_id: str, store: StoreDep) -> None:
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
