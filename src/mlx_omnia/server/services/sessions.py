"""The conversations the app keeps between runs.

What a message *is* stays the client's: the column holds the JSON array whole, and the only
thing checked on the way in is that it is an array. Nothing here renders these messages.

`updated_at` is the sidebar's order, so every write bumps it — the rename included.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter

from mlx_omnia.server.db.models.sessions import Session as SessionRow

_MESSAGES = TypeAdapter(list[JsonValue])
"""Both directions across the TEXT column. Reading is a parse and not a cast: a row somebody
hand-edited into the file fails here, named, instead of reaching a client as a string where
it expects a list."""


class NoSuchSession(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"no session {session_id!r}")
        self.session_id = session_id


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


@dataclass(frozen=True)
class SessionSummary:
    """What the list answers: the same row without its conversation."""

    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    message_count: int


def _view(row: SessionRow) -> SessionView:
    return SessionView(
        id=row.id,
        title=row.title,
        model=row.model,
        created_at=row.created_at,
        updated_at=row.updated_at,
        messages=_MESSAGES.validate_json(row.messages),
    )


def _summary(row: SessionRow) -> SessionSummary:
    return SessionSummary(
        id=row.id,
        title=row.title,
        model=row.model,
        created_at=row.created_at,
        updated_at=row.updated_at,
        message_count=len(_MESSAGES.validate_json(row.messages)),
    )


async def _found(session_id: str) -> SessionRow:
    row = await SessionRow.objects.get_or_none(id=session_id)
    if row is None:
        raise NoSuchSession(session_id)
    return row


async def summaries() -> list[SessionSummary]:
    """Most recently touched first, which is the order the sidebar draws."""
    rows = await SessionRow.objects.order_by(["-updated_at", "-id"]).all()
    return [_summary(row) for row in rows]


async def create(title: str, model: str) -> SessionView:
    """The id is the server's: a client that generated its own could collide with a second
    one on the same file, and the write would overwrite a conversation instead of failing."""
    now = time.time()
    row = SessionRow(
        id=uuid4().hex,
        title=title,
        model=model,
        created_at=now,
        updated_at=now,
        messages="[]",
    )
    await row.save()
    return _view(row)


async def view(session_id: str) -> SessionView:
    return _view(await _found(session_id))


async def update(session_id: str, title: str | None, model: str | None) -> SessionView:
    """A field left out goes on holding what the row already says."""
    row = await _found(session_id)
    await row.update(
        title=row.title if title is None else title,
        model=row.model if model is None else model,
        updated_at=time.time(),
    )
    return _view(row)


async def replace_messages(session_id: str, messages: list[JsonValue]) -> SessionView:
    """The whole array, not an append: what the client holds is the conversation, and a turn
    edited or dropped on that side has to be able to leave the file too."""
    row = await _found(session_id)
    await row.update(
        updated_at=time.time(),
        messages=_MESSAGES.dump_json(messages).decode(),
    )
    return _view(row)


async def remove(session_id: str) -> None:
    if not await SessionRow.objects.delete(id=session_id):
        raise NoSuchSession(session_id)
