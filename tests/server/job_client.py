"""A `TestClient` over the routes a job suite needs, wired the way `main` wires them.

The registry is no longer handed a store: the rows are ormar's and the file is the one
`mlx_omnia.paths` names, which the harness gives every test fresh. So the lifespan here is
`create_app`'s, minus everything a job suite does not touch — migrations, the connection, and
the single writer the registry drains its frames through.
"""

import asyncio
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from mlx_omnia.server.db import base as db
from mlx_omnia.server.main import migrate
from mlx_omnia.server.services.jobs import Jobs


def job_client(*routers: APIRouter) -> Iterator[TestClient]:
    registry = Jobs()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await asyncio.to_thread(migrate)
        await db.connect()
        await registry.open()
        yield
        await registry.close()
        await db.disconnect()

    app = FastAPI(lifespan=lifespan)
    for router in routers:
        app.include_router(router)
    app.state.jobs = registry
    with TestClient(app) as running:
        yield running
