"""A real server for `/api/anthropic/v1/*`, the SDK pointed at it, and the helpers every
suite over it reads answers with.

The server is a real one, like the metrics suite: `TestClient` and `ASGITransport` run the
whole response before handing it over, and both the SDK's stream and the ping the suites
assert about are about frames arriving while the generation is still going.
"""

import importlib
import json
import shutil
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass

import anthropic
import httpx
import pytest
import uvicorn
from anthropic.types import Message as Reply
from anthropic.types import MessageParam, TextBlockParam

from mlx_omnia import paths
from mlx_omnia.server.api.anthropic.models import MessagesRequest
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.profiles import Sampling
from tests.server.anthropic_doubles import Recording, fake_hub, load
from tests.server.anthropic_script import BUDGET, CATALOGUED, ECHO, PRESET
from tests.server.conftest import app_of, seed_config


@dataclass(frozen=True)
class Stand:
    base_url: str
    url: str
    """`/api/anthropic/v1/messages`, which the raw-frame tests post to directly."""
    engine: Recording


@pytest.fixture(autouse=True)
def fresh_state() -> None:
    """The harness wipes the state directory before every test; this stand is module-scoped
    and holds the database open, so the wipe happens once, with the stand."""


def seed_profiles() -> None:
    """The two profiles the suites select by `model:profile`. Written straight into the table
    the way `seed_config` writes config: the app is not up yet, and what the profile service
    would add on top of an insert is what the routes below read back anyway."""
    empty = Sampling().model_dump_json(exclude_none=True)
    with closing(sqlite3.connect(paths.server_db())) as connection, connection:
        connection.executemany(
            "INSERT INTO profiles(model, name, sampling, system_prompt, template, features)"
            " VALUES(?, ?, ?, ?, NULL, '{}')",
            [(ECHO, "terse", empty, PRESET), (CATALOGUED, "code", empty, None)],
        )


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    root = tmp_path_factory.mktemp("anthropic")
    fake_hub(root / "hub")
    shutil.rmtree(paths.state_dir(), ignore_errors=True)
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    seed_config({})
    seed_profiles()
    daemon = Daemon()
    engine = Recording(load, daemon, Metrics())
    # The daemon's own app: the handler that answers a refused body in this dialect is
    # registered by `create_app` and picks its encoder by route prefix, so an app with the
    # router mounted by hand would answer FastAPI's 422 and the SDK would raise the wrong
    # class for the wrong reason. That `create_app` registers it at all is
    # `test_dialect_errors.py`'s.
    app = app_of(engine, daemon)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # The catalog reads the machine's real Hugging Face cache: patched for the whole module,
    # so the listing test answers about this fixture and touches nothing of the user's. Both
    # the package's names and the scanner's own — the package re-exports the paths as values,
    # and `scan()` reads its own module's.
    scanner = importlib.import_module("mlx_omnia.server.services.catalog.scan")
    with pytest.MonkeyPatch.context() as patched:
        for module in (catalog, scanner):
            patched.setattr(module, "HUB_CACHE", root / "hub")
            patched.setattr(module, "QUANTIZED_CACHE", root / "quantized")
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        base_url = f"http://127.0.0.1:{port}"
        yield Stand(
            base_url=base_url,
            url=f"{base_url}/api/anthropic/v1/messages",
            engine=engine,
        )
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive(), "the stand's server did not shut down"


@pytest.fixture(scope="module")
def client(stand: Stand) -> anthropic.Anthropic:
    """No retries: the 500 one test provokes on purpose would otherwise be generated three
    times before the test that asked for it ever sees it."""
    return anthropic.Anthropic(
        base_url=f"{stand.base_url}/api/anthropic", api_key="unused", max_retries=0, timeout=60
    )


def turns(prompt: str) -> list[MessageParam]:
    return [{"role": "user", "content": prompt}]


def ask(
    client: anthropic.Anthropic,
    prompt: str,
    *,
    model: str = ECHO,
    max_tokens: int = BUDGET,
    system: str | list[TextBlockParam] | anthropic.Omit = anthropic.omit,
) -> Reply:
    return client.messages.create(
        model=model, messages=turns(prompt), max_tokens=max_tokens, system=system
    )


def only_text(reply: Reply) -> str:
    assert len(reply.content) == 1, f"expected one block, got {reply.content!r}"
    block = reply.content[0]
    assert block.type == "text", f"expected a text block, got {block.type!r}"
    return block.text


def entry(value: object) -> dict[str, object]:
    assert isinstance(value, dict), f"expected an object, got {value!r}"
    return value


def text(value: object) -> str:
    assert isinstance(value, str), f"expected a string, got {value!r}"
    return value


def envelope(body: object) -> tuple[str, str]:
    """The `type` and the `message` of the dialect's error body. The SDK has already chosen
    the exception class by status and hands this over raw — it is what a client's own error
    mapping reads, so it is what a test about the envelope has to open."""
    shape = entry(body)
    assert shape["type"] == "error", f"not the dialect's envelope: {shape!r}"
    error = entry(shape["error"])
    return text(error["type"]), text(error["message"])


def frames(stand: Stand, **body: object) -> list[tuple[str, dict[str, object]]]:
    """One `(event, payload)` per frame of a streamed request, in the order they arrived."""
    asked: dict[str, object] = {
        "model": ECHO,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": BUDGET,
        "stream": True,
    }
    captured: list[tuple[str, dict[str, object]]] = []
    named = ""
    with (
        httpx.Client() as http,
        http.stream("POST", stand.url, json=asked | body, timeout=60) as response,
    ):
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if line.startswith("event: "):
                named = line.removeprefix("event: ")
            elif line.startswith("data: "):
                captured.append((named, entry(json.loads(line.removeprefix("data: ")))))
    return captured


def body(**fields: object) -> MessagesRequest:
    asked: dict[str, object] = {
        "model": ECHO,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
    }
    return MessagesRequest.model_validate(asked | fields)
