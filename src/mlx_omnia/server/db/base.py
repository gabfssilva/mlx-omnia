"""The database every model in `db.models` shares.

One file, the same one the flat store wrote: `mlx_omnia.paths.server_db()`. WAL plus the
busy timeout is what lets a second writer wait instead of failing.
"""

from __future__ import annotations

import ormar
import sqlalchemy

from mlx_omnia import paths


def url() -> str:
    path = paths.server_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path}"


metadata = sqlalchemy.MetaData()
database = ormar.DatabaseConnection(url(), connect_args={"timeout": 5.0})
ormar_config = ormar.OrmarConfig(metadata=metadata, database=database)


async def connect() -> None:
    """Open the engine and switch the file to WAL. `journal_mode` is persistent and
    `foreign_keys` is set per-connection by ormar's own sqlite listener; the busy
    timeout rides in `connect_args`."""
    await database.connect()
    async with database.get_query_executor() as executor:
        await executor.execute(sqlalchemy.text("PRAGMA journal_mode = WAL"))


async def disconnect() -> None:
    if database.is_connected:
        await database.disconnect()


def create_all() -> None:
    engine = sqlalchemy.create_engine(f"sqlite:///{paths.server_db()}")
    metadata.create_all(engine)
    engine.dispose()
