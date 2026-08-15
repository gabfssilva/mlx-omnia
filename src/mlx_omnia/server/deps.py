"""The two things off the app every route reaches for: the engine and the daemon's database.

One home rather than a copy per router. A dependency is not a dialect's property, and ten
identical `_engine` functions are ten callables FastAPI resolves separately — the same
object either way, and one more place for the assertion to drift."""

from typing import Annotated

from fastapi import Depends, Request

from mlx_omnia.server.engine import Engine
from mlx_omnia.server.store import Store


async def engine_of(request: Request) -> Engine:
    """Async so it reads the engine's dicts on the loop that mutates them."""
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


def store_of(request: Request) -> Store:
    """Sync: the store is SQLite, and a route that touches it runs in the threadpool."""
    store = request.app.state.store
    assert isinstance(store, Store)
    return store


EngineDep = Annotated[Engine, Depends(engine_of)]
StoreDep = Annotated[Store, Depends(store_of)]
