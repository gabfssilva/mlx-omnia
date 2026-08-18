"""The two reads the engine's environment makes from threads the async pool cannot serve.

The engine calls its `Environment` from the loop, from worker threads and from inside the
decode loop, so these are plain `sqlite3` SELECTs per call — the same cost and the same
freshness the old store gave those paths. Everything else in the server reads through ormar;
anything that grows here beyond a SELECT belongs there instead.
"""

import sqlite3
from contextlib import closing

from mlx_omnia import paths

_BUSY_SECONDS = 5.0


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(paths.server_db(), timeout=_BUSY_SECONDS)
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def config_values() -> dict[str, str]:
    with closing(_connect()) as connection:
        try:
            rows = connection.execute("SELECT key, value FROM config").fetchall()
        except sqlite3.OperationalError:
            # No table yet: a boot before the first migration ran answers the defaults.
            return {}
    return {str(key): str(value) for key, value in rows}


def model_settings(model_id: str) -> tuple[str, int | None]:
    """`(features, max_concurrent_requests)`, with the empty defaults an absent row means."""
    with closing(_connect()) as connection:
        try:
            row = connection.execute(
                "SELECT features, max_concurrent_requests FROM model_settings WHERE model = ?",
                (model_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return "{}", None
    if row is None:
        return "{}", None
    features, limit = row
    return str(features), int(limit) if limit is not None else None
