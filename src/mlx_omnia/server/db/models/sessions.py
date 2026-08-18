"""The `sessions` table: a conversation as the file holds it."""

from __future__ import annotations

import ormar

from mlx_omnia.server.db import base


class Session(ormar.Model):
    """`messages` is JSON text and stays opaque: what a message is belongs to the dialect,
    and a table under it would make every new content part a migration."""

    ormar_config = base.ormar_config.copy(tablename="sessions")

    id: str = ormar.Text(primary_key=True)
    title: str = ormar.Text()
    model: str = ormar.Text(default="")
    created_at: float = ormar.Float()
    updated_at: float = ormar.Float()
    messages: str = ormar.Text(default="[]")
