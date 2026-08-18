"""The `jobs` table: what is left of a job once its stream is gone."""

from __future__ import annotations

from typing import Literal

import ormar

from mlx_omnia.server.db import base

JobState = Literal["pending", "running", "ok", "error", "cancelled"]

JOB_STATES: tuple[JobState, ...] = ("pending", "running", "ok", "error", "cancelled")


class Job(ormar.Model):
    """`progress` is the last frame as JSON text, so a client arriving after a restart
    still sees where the job stopped. `subject` is what the job is about, as JSON text."""

    ormar_config = base.ormar_config.copy(tablename="jobs")

    id: str = ormar.Text(primary_key=True)
    kind: str = ormar.Text()
    subject: str = ormar.Text(default="{}")
    state: str = ormar.Text(choices=list(JOB_STATES))
    progress: str = ormar.Text()
    created_at: float = ormar.Float()
    updated_at: float = ormar.Float()
    error: str | None = ormar.Text(nullable=True)
