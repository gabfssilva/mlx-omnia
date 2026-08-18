"""What every route reaches for off the app: the engine, the job registry, the bind host.

One home rather than a copy per router — the same object either way, and one more place for
the assertion to drift.
"""

from typing import Annotated

from fastapi import Depends, Request

from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services.jobs import Jobs


async def engine_of(request: Request) -> Engine:
    """Async so it reads the engine's dicts on the loop that mutates them."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


async def jobs_of(request: Request) -> Jobs:
    registry = request.app.state.jobs
    assert isinstance(registry, Jobs)
    return registry


async def host_of(request: Request) -> str:
    """What the socket was bound to: off the loopback the api key cannot be cleared."""
    host = request.app.state.host
    assert isinstance(host, str)
    return host


EngineDep = Annotated[Engine, Depends(engine_of)]
JobsDep = Annotated[Jobs, Depends(jobs_of)]
HostDep = Annotated[str, Depends(host_of)]
