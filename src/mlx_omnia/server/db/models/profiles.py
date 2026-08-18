"""The `config`, `model_settings` and `profiles` tables: what a request gets before it asks.

`sampling` and `features` are JSON text because their vocabulary belongs to the sampling
module and to the engine's switches; a column per knob would make every new one a migration.

`profiles` is keyed by `(model, name)` in the file, and ormar carries one primary key. It is
sqlite's own `rowid` rather than `model`: ormar folds rows that share a primary key into one,
so a model with two profiles would answer with whichever came last — the rows are all in the
file, and only the read loses them. `rowid` exists on every table here (none is `WITHOUT
ROWID`) and is unique by definition, which is what the fold needs to be a no-op. Writes still
filter on `(model, name)`, the key the file enforces.
"""

from __future__ import annotations

import ormar

from mlx_omnia.server.db import base


class Config(ormar.Model):
    """The daemon's settings, one row per key."""

    ormar_config = base.ormar_config.copy(tablename="config")

    key: str = ormar.Text(primary_key=True)
    value: str = ormar.Text()


class ModelSettings(ormar.Model):
    """What every request for a model gets before any profile opines. A knob left unset is
    one this daemon does not opine on, and the checkpoint's own file still answers for it."""

    ormar_config = base.ormar_config.copy(tablename="model_settings")

    model: str = ormar.Text(primary_key=True)
    features: str = ormar.Text(default="{}")
    max_concurrent_requests: int | None = ormar.Integer(nullable=True)
    sampling: str = ormar.Text(default="{}")


class Profile(ormar.Model):
    """One named preset over a model's settings."""

    ormar_config = base.ormar_config.copy(tablename="profiles")

    rowid: int | None = ormar.Integer(primary_key=True, autoincrement=True, nullable=True)
    model: str = ormar.Text()
    name: str = ormar.Text()
    sampling: str = ormar.Text()
    system_prompt: str | None = ormar.Text(nullable=True)
    template: str | None = ormar.Text(nullable=True)
    features: str = ormar.Text(default="{}")
