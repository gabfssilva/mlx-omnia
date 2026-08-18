"""The server the `/responses` suites are judged against: a real uvicorn process over the
scripted models, wired by `create_app` the way the daemon wires one.

A real process, because `TestClient` and `ASGITransport` run the whole response before handing
it over and half of what these suites assert is about *when* a frame arrives.
"""

import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import httpx
import mlx.core as mx
import pytest
import uvicorn
from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall

from mlx_omnia import Chat, ChatMessage
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.main import create_app
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Loader
from tests.server.responses_script import GUIDED, TEMPLATE, loader


class Free:
    """A walk that forbids nothing.

    No scripted model here holds a token table, so a real `Vocabulary` cannot be built over
    one and every strict request would end in the same refusal. What a constrained request
    has to prove on this stand is the wiring — that the route compiles the schema and hands
    the walk to the generation — and that is what this stands in for.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return logits

    def accept(self, token: int) -> bool:
        return True


class Constrained(Engine):
    """The engine with one constrainable model in it, and the schemas it was asked to compile.

    Only `GUIDED` gets the double: every other id falls through to the engine's own
    `constrain`, which is what makes the refusal below a real one — nothing under a scripted
    model has a tokenizer, a head width or a stop id to compile against.
    """

    def __init__(self, loader: Loader, daemon: Daemon, metrics: Metrics) -> None:
        super().__init__(loader, daemon, metrics)
        self.compiled: list[Mapping[str, object]] = []

    async def constrain(self, model_id: str, schema: Mapping[str, object]) -> Constraint:
        if model_id != GUIDED:
            return await super().constrain(model_id, schema)
        self.compiled.append(schema)
        return Free()


@dataclass
class Stand:
    base_url: str
    engine: Constrained


@pytest.fixture(scope="module")
def stand() -> Iterator[Stand]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    daemon = Daemon()
    engine = Constrained(loader, daemon, Metrics())
    app = create_app(engine, host="127.0.0.1", daemon=daemon)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield Stand(base_url=f"http://127.0.0.1:{port}", engine=engine)
    server.should_exit = True
    thread.join(timeout=5)
    assert not thread.is_alive(), "the stand's server did not shut down"


@pytest.fixture(scope="module")
def client(stand: Stand) -> OpenAI:
    return OpenAI(base_url=f"{stand.base_url}/api/openai/v1", api_key="unused")


def save_profile(stand: Stand, model: str, name: str, body: Mapping[str, object]) -> None:
    """A profile written the one way a client can write one. The store is gone and the service
    behind this route is async on the server's own loop, so the route is also the only way in
    from a test thread."""
    response = httpx.put(
        f"{stand.base_url}/admin/models/{model}/profiles/{name}", json=body, timeout=30
    )
    assert response.status_code == 200, response.text


def entry(value: object) -> dict[str, object]:
    assert isinstance(value, dict), f"expected an object, got {value!r}"
    return value


def code(body: object) -> str:
    """The dialect's own error envelope, read off the exception's raw body: the status alone
    does not say which of two client errors this was.

    `body` is already the inside of the envelope: the SDK builds the exception with
    `data.get("error", data)`, so what reaches here is what the server wrote under `error` —
    and unwrapping it a second time is how this read fails."""
    value = entry(body)["code"]
    assert isinstance(value, str), f"expected a code, got {value!r}"
    return value


def rendered(turns: tuple[ChatMessage, ...], tools: tuple[Mapping[str, object], ...] = ()) -> str:
    return TEMPLATE.render(Chat(turns, tools=tools))


def only_call(calls: Sequence[ResponseFunctionToolCall]) -> ResponseFunctionToolCall:
    """The one `function_call` item of an answer, with the two ids it has to carry: the item's
    own, which every frame about it repeats, and `call_id`, which is what the client sends
    back. A route that used one string for both would pass any assertion about either."""
    assert len(calls) == 1, f"expected one call, got {calls!r}"
    call = calls[0]
    assert call.name == "get_weather"
    assert call.id is not None and call.id.startswith("fc_")
    assert call.call_id.startswith("call_") and call.call_id != call.id
    return call
