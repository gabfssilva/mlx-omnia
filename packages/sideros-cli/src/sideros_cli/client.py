"""HTTP client for the sideros daemon. This package never imports the engine.

What this package exists to prove — `uv tree --package sideros-cli` showing httpx and its
transitives and nothing else — dies the moment it grows a dependency on the engine.

Every call answers with a value or raises `ServerError`, carrying the daemon's own sentence
and the status it came with — `None` when nothing answered at all, which is the daemon being
down and the one failure the CLI turns into a spawn instead of a message.

Routes `/admin` does not serve yet — pull, benches, config, metrics — are absent on purpose.
"""

import json
from collections.abc import Generator, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from typing import NotRequired, TypedDict, cast
from urllib.parse import quote

import httpx

PROBE_TIMEOUT = 2.0
"""`/admin/health`, which is asked before every command: it answers off a dict."""

READ_TIMEOUT = 10.0
"""The catalog and the system read: both stat the whole cache before answering."""

REMOVE_TIMEOUT = 60.0
"""A delete unlinks the repository's blobs, which is tens of gigabytes of unlink."""

STREAM_TIMEOUT = 120.0
"""Between two frames, not for the whole stream: what sits in the gap is a prefill, and the
server keeps the connection warm every half second while it runs."""


@dataclass(eq=False)
class ServerError(Exception):
    """Not frozen, unlike the app's copy of this class: `contextlib.contextmanager` assigns
    `__traceback__` on an exception that passes through it, a frozen dataclass refuses the
    assignment, and what reaches the user is a `FrozenInstanceError` naming neither the
    daemon nor the failure. `eq=False` keeps an exception hashable, which is the default
    everywhere else in the language."""

    message: str
    status: int | None = None
    """The HTTP status, or `None` when nothing answered — the daemon being down."""

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Health:
    status: str
    models: tuple[str, ...]
    """The ids the engine holds loaded: empty at boot, and a consequence of requests."""


@dataclass(frozen=True)
class SystemInfo:
    chip: str
    gpu_cores: int
    memory_bytes: int
    bandwidth_theoretical_gbs: float
    bandwidth_sustained_gbs: float
    disk_free_bytes: int
    catalog: str
    version: str


@dataclass(frozen=True)
class ModelEntry:
    id: str
    """What a request names, and what the delete route takes back."""
    architecture: str
    quantization: str | None
    """The label the config declares — `None` when it declares none, which is dense."""
    bytes_on_disk: int
    resident: bool


class _HealthJson(TypedDict):
    status: str
    models: list[str]


class _SystemJson(TypedDict):
    chip: str
    gpu_cores: int
    memory_bytes: int
    bandwidth_theoretical_gbs: float
    bandwidth_sustained_gbs: float
    disk_free_bytes: int
    catalog: str
    version: str


class _ModelJson(TypedDict):
    id: str
    architecture: str
    quantization: str | None
    bytes_on_disk: int
    resident: bool


class _DeltaJson(TypedDict):
    content: NotRequired[str]


class _ChoiceJson(TypedDict):
    delta: _DeltaJson


class _ChunkJson(TypedDict):
    choices: list[_ChoiceJson]


@dataclass(frozen=True)
class Event:
    name: str
    data: str


def _detail(response: httpx.Response) -> str:
    """The daemon's words out of the two error shapes it writes: FastAPI's `detail` and the
    dialect's `error.message`. The refusal to delete a resident model says why in there, and
    a status code alone would leave the user with nothing to act on."""
    try:
        body: object = response.json()
    except ValueError:
        return response.text.strip()
    if not isinstance(body, dict):
        return response.text.strip()
    fields = cast(dict[str, object], body)
    detail = fields.get("detail")
    if isinstance(detail, str):
        return detail
    error = fields.get("error")
    if isinstance(error, dict):
        message = cast(dict[str, object], error).get("message")
        if isinstance(message, str):
            return message
    return response.text.strip()


def _check(response: httpx.Response) -> None:
    if response.is_success:
        return
    raise ServerError(
        f"{response.request.url} answered {response.status_code}: {_detail(response)}",
        response.status_code,
    )


def _unreadable(response: httpx.Response, error: Exception) -> ServerError:
    return ServerError(
        f"{response.request.url} answered a body this client does not read: {error}",
        response.status_code,
    )


def _request(
    method: str,
    base_url: str,
    path: str,
    *,
    timeout: float,
    params: dict[str, str] | None = None,
) -> httpx.Response:
    """`InvalidURL` alongside the transport failures: the address is a flag the user types."""
    url = f"{base_url}{path}"
    try:
        response = httpx.request(method, url, params=params, timeout=timeout)
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        raise ServerError(f"{url} did not answer: {error}") from error
    _check(response)
    return response


def _path(model_id: str) -> str:
    """Ids carry slashes — a repository id, and a whole directory for a quantized entry —
    and the route that takes them is a `:path`, so the separators stay separators."""
    return f"/admin/models/{quote(model_id, safe='/')}"


def health(base_url: str, *, timeout: float = PROBE_TIMEOUT) -> Health:
    """Whether the daemon is up, and what it is holding in memory."""
    response = _request("GET", base_url, "/admin/health", timeout=timeout)
    try:
        body: _HealthJson = response.json()
        return Health(body["status"], tuple(body["models"]))
    except (ValueError, KeyError, TypeError) as error:
        raise _unreadable(response, error) from error


def system(base_url: str, *, timeout: float = READ_TIMEOUT) -> SystemInfo:
    """The machine, plus the two bandwidths every % of ceiling divides by."""
    response = _request("GET", base_url, "/admin/system", timeout=timeout)
    try:
        body: _SystemJson = response.json()
        return SystemInfo(
            chip=body["chip"],
            gpu_cores=body["gpu_cores"],
            memory_bytes=body["memory_bytes"],
            bandwidth_theoretical_gbs=body["bandwidth_theoretical_gbs"],
            bandwidth_sustained_gbs=body["bandwidth_sustained_gbs"],
            disk_free_bytes=body["disk_free_bytes"],
            catalog=body["catalog"],
            version=body["version"],
        )
    except (ValueError, KeyError, TypeError) as error:
        raise _unreadable(response, error) from error


def catalog(
    base_url: str, *, resident: bool = False, timeout: float = READ_TIMEOUT
) -> list[ModelEntry]:
    """The models on disk. `resident=True` narrows it to the ones loaded right now."""
    params = {"resident": "true"} if resident else None
    response = _request("GET", base_url, "/admin/models", timeout=timeout, params=params)
    try:
        rows: list[_ModelJson] = response.json()
        return [
            ModelEntry(
                id=row["id"],
                architecture=row["architecture"],
                quantization=row["quantization"],
                bytes_on_disk=row["bytes_on_disk"],
                resident=row["resident"],
            )
            for row in rows
        ]
    except (ValueError, KeyError, TypeError) as error:
        raise _unreadable(response, error) from error


def delete_model(base_url: str, model_id: str, *, timeout: float = REMOVE_TIMEOUT) -> None:
    """Removes it from disk. The server refuses while the model is resident, and the refusal
    arrives as the reason it was refused."""
    _request("DELETE", base_url, _path(model_id), timeout=timeout)


def read_events(lines: Iterable[str]) -> Iterator[Event]:
    """SSE frames out of raw lines: a frame closes at the blank line, `data:` accumulates
    until then, and a line opening with `:` is the keep-alive the server sends through a long
    prefill."""
    name = "message"
    data: list[str] = []
    for line in lines:
        if line:
            field, _, value = line.partition(":")
            if field == "data":
                data.append(value.removeprefix(" "))
            elif field == "event":
                name = value.removeprefix(" ")
            continue
        if data:
            yield Event(name, "\n".join(data))
        name, data = "message", []


def sse(
    base_url: str,
    path: str,
    body: dict[str, object],
    *,
    timeout: float = STREAM_TIMEOUT,
) -> Generator[Event]:
    """A generator rather than an iterator in the signature: closing it is the operation the
    caller needs, and it is what ends the request."""
    try:
        client = httpx.Client(timeout=timeout)
        with client, client.stream("POST", f"{base_url}{path}", json=body) as response:
            if not response.is_success:
                # An error body is not streamed, and it is the whole message.
                response.read()
                _check(response)
            yield from read_events(response.iter_lines())
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        raise ServerError(f"{base_url}{path} did not answer: {error}") from error


def stream_chat(
    base_url: str,
    model: str,
    messages: list[Message],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float = STREAM_TIMEOUT,
) -> Generator[str]:
    """Yields assistant text pieces from the server's SSE stream.

    The inner generator is closed explicitly rather than left to the collector: closing this
    one is how Ctrl-C reaches the daemon — the connection drops, the daemon's stream ends and
    it cancels the job — and a chain that only unwinds when the object is collected would
    make that depend on when the collector runs.
    """
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    events = sse(base_url, "/api/openai/v1/chat/completions", body, timeout=timeout)
    with closing(events):
        for event in events:
            if event.data == "[DONE]":
                return
            try:
                chunk: _ChunkJson = json.loads(event.data)
                piece = chunk["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise ServerError(
                    f"the chat stream sent a chunk this client does not read: {error}"
                ) from error
            if piece:
                yield piece
