"""Alembic environment for the daemon's single sqlite file.

The url is not read from the ini file: where the state lives is `mlx_omnia.paths`' answer, and
a second copy of it in configuration would be a second answer.
"""

from __future__ import annotations

import sqlalchemy
from alembic import context

from mlx_omnia import paths
from mlx_omnia.server.db.base import metadata


def _url() -> str:
    path = paths.server_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = sqlalchemy.create_engine(_url())
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
