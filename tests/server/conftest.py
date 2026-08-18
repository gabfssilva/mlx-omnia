"""The server suite's harness.

The daemon reads one state directory (`mlx_omnia.paths`), and the database module binds its
url the moment it is imported — so the override below runs before anything under
`mlx_omnia.server` does, unconditionally: inheriting a real `OMNIA_STATE_DIR` here would
point the wipe fixture at somebody's actual state.
"""

import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing

os.environ["OMNIA_STATE_DIR"] = tempfile.mkdtemp(prefix="omnia-tests-")

import pytest
from fastapi import FastAPI

from mlx_omnia import paths
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.main import create_app, migrate
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Loader


@pytest.fixture(autouse=True)
def fresh_state() -> None:
    """Every test starts from an empty state directory — the per-test `tmp_path` the old
    `Store(tmp_path / "server.db")` gave each test, recreated for the one shared path."""
    root = paths.state_dir()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def seed_config(values: dict[str, object]) -> None:
    """Rows in `config` before the app boots, as the values themselves — the table stores
    JSON, so `64` is a number and `"secret"` a string."""
    migrate()
    with closing(sqlite3.connect(paths.server_db())) as connection, connection:
        connection.executemany(
            "INSERT INTO config(key, value) VALUES(?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(key, json.dumps(value)) for key, value in values.items()],
        )


def engine_of(loader: Loader) -> Engine:
    """An engine wired the way `main` wires one: the daemon as its environment, a real
    register as its sink — `create_app` asserts the latter."""
    return Engine(loader, Daemon(), Metrics())


def app_of(engine: Engine, daemon: Daemon | None = None) -> FastAPI:
    """Pass the same `Daemon` the engine was built over when a test exercises the disk
    tier — the lifespan binds the loop its vaults write through."""
    return create_app(engine, host="127.0.0.1", daemon=daemon)


def wired(loader: Loader) -> FastAPI:
    """The whole default wiring in one call, for tests that never touch the engine."""
    daemon = Daemon()
    return create_app(Engine(loader, daemon, Metrics()), host="127.0.0.1", daemon=daemon)
