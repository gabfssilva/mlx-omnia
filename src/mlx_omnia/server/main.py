"""The app factory and the daemon's entry point.

Everything above the engine is wired here: the database is opened and migrated in the
lifespan, the routers are mounted in the order their paths need, and the engine is handed the
one port through which it reads the daemon's settings.
"""

import argparse
import asyncio
import os
import signal
import threading
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlx_omnia import paths
from mlx_omnia.server import events
from mlx_omnia.server import metrics as metrics_module
from mlx_omnia.server.api import anthropic, auth, errors, gemini, management, openai_chat, responses
from mlx_omnia.server.daemon import Daemon, config_now, resident_loader
from mlx_omnia.server.db import base as db
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services import catalog, prefixes
from mlx_omnia.server.services.jobs import Jobs

_LIVE_BEAT_SECONDS = 0.5
"""How often the request being served is republished. Fast enough that a decode rate reads as
a rate, slow enough that a screen is not redrawn per token."""


def migrate() -> None:
    """Bring the file up to head. A database the flat store built is already at head, which
    the first revision recognises and skips."""
    ini = Path(__file__).parent / "db" / "migrations" / "alembic.ini"
    command.upgrade(Config(str(ini)), "head")


async def _beating(register: metrics_module.Metrics) -> None:
    """Nothing calls the register while a request runs, so nothing would say how it is
    going. This is the one thing in the daemon on a clock rather than an edge."""
    while True:
        await asyncio.sleep(_LIVE_BEAT_SECONDS)
        register.beat()


async def _invalid_request(request: Request, error: Exception) -> JSONResponse:
    """A body no dialect's schema accepts, in the dialect of whoever sent it.

    The SDKs do two different things with an error body: OpenAI's and Anthropic's hand it over
    raw, Gemini's parses it and prints `None None.` for somebody else's envelope. `/admin` and
    anything unrouted get OpenAI's.
    """
    assert isinstance(error, RequestValidationError)
    # The deepest error and not the first: a field written inside a union fails once per
    # member, and the first is whichever member pydantic tried first.
    deepest = max(error.errors(), key=lambda entry: len(entry["loc"]))
    location = [str(part) for part in deepest["loc"][1:] if "[" not in str(part)]
    message = f"{'.'.join(location) or 'body'}: {deepest['msg']}"
    path = request.url.path
    if path.startswith("/api/anthropic/"):
        return errors.anthropic_error(400, "invalid_request_error", message)
    if path.startswith("/api/gemini/"):
        return errors.gemini_error("INVALID_ARGUMENT", message)
    return errors.openai_error(400, message, "invalid_request")


def create_app(
    engine: Engine,
    *,
    host: str = "127.0.0.1",
    daemon: Daemon | None = None,
) -> FastAPI:
    """`host` is what the socket will be bound to, and it is here because one route needs it:
    off the loopback the api key is the only thing between the weights and the network, so
    `/admin/config` refuses to clear it. `daemon` is the engine's environment when there is
    one, and it is passed so the lifespan can give it the loop its vaults write through.
    """

    registry = Jobs()
    register = engine.metrics
    assert isinstance(register, metrics_module.Metrics)

    async def _health() -> object:
        return await management.health(engine)

    async def _config() -> object:
        return await management.config()

    async def _models() -> object:
        return await management.models(engine)

    async def _state() -> object:
        return await management.state(engine)

    async def _jobs() -> object:
        """The live ones. The history is a list that only grows and only one screen reads."""
        return await management.jobs(active=True)

    async def _metrics() -> object:
        return register.snapshot()

    hub = events.Hub(
        {
            # The order the opening burst arrives in, which is the order a screen needs it:
            # what the daemon is, then what it was told, then what it has.
            "health": _health,
            "config": _config,
            "models": _models,
            "state": _state,
            "jobs": _jobs,
            "metrics": _metrics,
        }
    )
    engine.on_change = lambda: hub.publish("state")
    register.on_change = lambda: hub.publish("metrics")
    registry.on_change = lambda: hub.publish("jobs")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(migrate)
        await db.connect()
        if daemon is not None:
            daemon.bind(loop)
        hub.bind(loop)
        beat = asyncio.create_task(_beating(register))
        watcher = events.observe(hub, (catalog.HUB_CACHE, catalog.QUANTIZED_CACHE))
        # Rows whose file is gone — a cache directory somebody cleared, a disk reformatted
        # under a database that was not. Left in, the ceiling would be enforced against bytes
        # nobody holds.
        await prefixes.sweep()
        await registry.open()
        engine.start()
        yield
        await engine.stop()
        await registry.close()
        beat.cancel()
        events.shutdown(watcher)
        await db.disconnect()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.host = host
    app.state.jobs = registry
    app.state.metrics = register
    app.state.events = hub
    # Ahead of every route, including the ones no router claims: an unrouted path under a
    # dialect prefix answers 401 rather than a 404 that enumerates what is mounted.
    app.add_middleware(auth.ApiKey)
    if os.environ.get("OMNIA_DEV") == "1":
        # Outside the key check, so a stream that was refused is a refusal on stdout too.
        app.add_middleware(events.Tap)
    app.include_router(openai_chat.router)
    app.include_router(responses.router)
    app.include_router(anthropic.router)
    app.include_router(gemini.router)
    # Last: its catalog routes end in `{model_id:path}`, which matches slashes.
    app.include_router(management.router)
    app.add_exception_handler(RequestValidationError, _invalid_request)
    return app


def watch_parent(pid: int, gone: Callable[[], None], every: float = 1.0) -> None:
    """Call `gone` once the process that asked to be the owner is no longer there.

    Signal 0 asks the kernel whether the pid exists without sending anything. Only
    `ProcessLookupError` means gone — a pid that exists but belongs to someone else raises
    `PermissionError`, and killing on that would be killing on a permission answer.
    """

    def watch() -> None:
        while True:
            time.sleep(every)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                gone()
                return

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(prog="omnia-server")
    parser.add_argument("--host", default="127.0.0.1")
    # The process this daemon belongs to: when it dies, so does the daemon. Off by default —
    # a daemon spawned by the CLI is meant to outlive the command that started it.
    parser.add_argument("--parent-pid", type=int, default=None)
    # No default, so that the saved port is what a bare `omnia-server` binds and an explicit
    # `--port` still wins.
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # Before the first read, which would otherwise create an empty database beside the one
    # holding the user's bench history and leave it stranded at the old address.
    if (former := paths.adopt_former_state()) is not None:
        print(f"moved state from {former} to {paths.state_dir()}")
    booted = config_now()
    auth.check_bind(args.host, booted.api_key)
    if args.parent_pid is not None:
        # SIGTERM to self rather than a direct exit: it is the same shutdown the owner's own
        # kill takes, so uvicorn drains what is open instead of dropping it mid-stream.
        watch_parent(args.parent_pid, lambda: os.kill(os.getpid(), signal.SIGTERM))
    port = booted.port if args.port is None else args.port
    daemon = Daemon()
    engine = Engine(resident_loader(), daemon, metrics_module.Metrics())
    app = create_app(engine, host=args.host, daemon=daemon)
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
