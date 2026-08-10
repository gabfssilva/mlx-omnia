"""`GET /admin/events`: everything the app draws, pushed once per change.

One stream and not one per resource, because the app opens all of them at once and a
browser spends its connections on nothing else. Each frame names the resource it carries —
`{"resource": "models", "value": ...}` — so a client dispatches on the name and a change to
the queue does not resend the catalog.

The hub carries *names* and never values: a watcher marks the resource dirty and the
generator reads it when it drains. That is what makes a burst cheap — two hundred writes
landing while the catalog is being built collapse into the one rebuild that follows, and
the coalescing costs nothing to arrange because it is what a set does.

What raises a name is whoever moved the thing: the engine on a load and on the queue, the
job registry on a report, the metrics register per request, `PATCH /admin/config` on the
way out, and — for the one resource nothing in this process owns — FSEvents over the two
checkpoint caches. The disk is the only publisher that needs a delay in front of it: a
download writes at disk speed and every chunk is an event, so `_Disk` sits on them for a
window and raises the name once.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

RESOURCES = ("health", "config", "models", "state", "jobs", "metrics")
"""Everything the stream carries, in the order a subscriber is opened with: what the daemon
is, then what it was told, then what it has. A client draws every screen off these six and
asks for nothing else on a timer."""

Source = Callable[[], Awaitable[object]]
"""How a resource answers. Async because half of them read state the loop owns."""

_KEEP_ALIVE_SECONDS = 15.0

_DISK_WINDOW_SECONDS = 0.4
"""How long FSEvents chatter is held before the catalog is named. A download of one shard
is thousands of events and one change; a delete is a handful and one change."""


@dataclass(eq=False)
class _Watcher:
    """`eq=False` so two subscribers with the same pending set are still two subscribers:
    the hub holds them in a set, and identity is what a subscription is."""

    dirty: set[str] = field(default_factory=set)
    wake: asyncio.Event = field(default_factory=asyncio.Event)


class Hub:
    """The fan-out. `publish` is callable from any thread — FSEvents has its own, and so
    does every job body — and delivery always lands on the loop the app runs on."""

    def __init__(self, sources: Mapping[str, Source]) -> None:
        assert tuple(sources) == RESOURCES, f"the hub answers for {RESOURCES}, not {tuple(sources)}"
        self.sources = sources
        self._watchers: set[_Watcher] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Taken at startup and not at construction: `create_app` builds the hub before
        there is a loop to run it on."""
        self._loop = loop

    def publish(self, resource: str) -> None:
        if self._loop is None:
            # Before startup and after shutdown. Nobody is watching in either, and a frame
            # for a resource that will be read whole by the next subscriber is not owed.
            return
        self._loop.call_soon_threadsafe(self._mark, resource)

    def _mark(self, resource: str) -> None:
        for watcher in self._watchers:
            watcher.dirty.add(resource)
            watcher.wake.set()

    @contextmanager
    def watch(self) -> Generator[_Watcher]:
        watcher = _Watcher()
        self._watchers.add(watcher)
        try:
            yield watcher
        finally:
            self._watchers.discard(watcher)


class _Disk(FileSystemEventHandler):
    """FSEvents over the checkpoint caches, folded into one delayed `models`.

    The handler runs on watchdog's thread and touches nothing but the hub, which is the
    only object here that is safe to call from one."""

    def __init__(self, hub: Hub, loop: asyncio.AbstractEventLoop) -> None:
        self._hub = hub
        self._loop = loop
        self._timer: asyncio.TimerHandle | None = None

    def on_any_event(self, event: FileSystemEvent) -> None:
        self._loop.call_soon_threadsafe(self._defer)

    def _defer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self._loop.call_later(_DISK_WINDOW_SECONDS, self._raise)

    def _raise(self) -> None:
        self._timer = None
        self._hub.publish("models")


def observe(hub: Hub, roots: tuple[Path, ...]) -> BaseObserver:
    """Watches whichever of the caches exist. A root that is not there yet is not watched
    and not an error: the quantized cache appears the first time something is quantized,
    and what makes it appear is a job that names the catalog itself."""
    observer = Observer()
    handler = _Disk(hub, asyncio.get_running_loop())
    for root in roots:
        if root.is_dir():
            observer.schedule(handler, str(root), recursive=True)
    observer.start()
    return observer


def _frame(resource: str, value: object) -> str:
    return f"data: {json.dumps({'resource': resource, 'value': jsonable_encoder(value)})}\n\n"


async def _frames(hub: Hub) -> AsyncIterator[str]:
    """Subscribed before the first resource is read, so a change that lands during the
    opening burst is delivered after it instead of being missed.

    The stream ends when the client goes away and never on its own — there is no last
    frame, the way there is no last state."""
    with hub.watch() as watcher:
        for resource, source in hub.sources.items():
            yield _frame(resource, await source())
        while True:
            try:
                await asyncio.wait_for(watcher.wake.wait(), _KEEP_ALIVE_SECONDS)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            watcher.wake.clear()
            pending, watcher.dirty = watcher.dirty, set()
            for resource in sorted(pending):
                # Everything raised while the previous resource was being built is in
                # `dirty` again by now, which is the coalescing: one rebuild per drain.
                yield _frame(resource, await hub.sources[resource]())


class Tap:
    """Every line of every `text/event-stream` under `/admin`, on stdout.

    Pure ASGI for the reason `auth.ApiKey` is: it wraps the bytes on their way out and puts
    nothing between the generator and the socket. What is tapped is decided by the content
    type and not by the route, so a stream added later is covered without being named here —
    `/admin/events`, the per-job streams and the metrics register alike.

    The dialects are out, and by prefix rather than by a list of paths: every token stream
    this daemon serves is under `/api`, and a fourth dialect is a fourth prefix nobody has
    to remember to exclude. Two reasons, and the second is the one that matters. A
    generation is the one stream whose lines are the answer somebody is reading on screen
    already. And the write here is synchronous and flushed per line, on the loop that is
    carrying those tokens — a debugging tool must not be in the path of the thing this
    daemon exists to be fast at.

    A frame split across two chunks is held and printed whole: what a reader wants is the
    line the server wrote, not the packet it happened to fit in.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/admin/"):
            await self._app(scope, receive, send)
            return
        streaming = False
        held = b""

        async def tapped(message: Message) -> None:
            nonlocal streaming, held
            if message["type"] == "http.response.start":
                streaming = any(
                    name.lower() == b"content-type" and b"text/event-stream" in value
                    for name, value in message["headers"]
                )
            elif streaming and message["type"] == "http.response.body":
                held += message.get("body", b"")
                *lines, held = held.split(b"\n")
                for line in lines:
                    print(f"sse {scope['path']} | {line.decode(errors='replace')}", flush=True)
            await send(message)

        await self._app(scope, receive, tapped)


def announce(request: Request, resource: str) -> None:
    """What a handler calls when it changed something no object owns — the config is a row
    in the store and nothing watches the store. Tolerant of an app with no hub, which is
    every test that mounts one router: a change nobody is streaming is not an error."""
    hub = getattr(request.app.state, "events", None)
    if isinstance(hub, Hub):
        hub.publish(resource)


def _hub(request: Request) -> Hub:
    hub = request.app.state.events
    assert isinstance(hub, Hub)
    return hub


router = APIRouter()


@router.get("/admin/events")
async def events(request: Request) -> StreamingResponse:
    return StreamingResponse(_frames(_hub(request)), media_type="text/event-stream")


def shutdown(observer: BaseObserver) -> None:
    """Joined and not just stopped: watchdog's thread is not a daemon, and a process that
    leaves it running does not exit."""
    observer.stop()
    with suppress(RuntimeError):
        observer.join()
