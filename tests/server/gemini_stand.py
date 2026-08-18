"""The stand the Gemini suites share: the doubles, the fixtures, and the constants.

The judge is the official `google-genai` SDK because this dialect's client does more than read
a status: it *parses* the error body, and it accumulates a stream whose frames it decodes one
by one. `TestClient` and `ASGITransport` run a response to completion before handing it over,
so the stream would be judged after the fact — the server here is a real one on a real port,
like the jobs and metrics suites.

The model under the engine answers with the prompt the checkpoint's own template rendered.
That is the only place the conversion this stage owns is visible: which turn each `content`
became, where a `systemInstruction` went and which name resolved to a checkpoint all exist
inside the render, and the render is on the other side of `stream`.
"""

import importlib
import shutil
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass

import pytest
import uvicorn
from google import genai
from google.genai import types

from mlx_omnia import Chat
from mlx_omnia import ChatMessage as Turn
from mlx_omnia.paths import server_db, state_dir
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.profiles import Sampling
from tests.server.conftest import app_of
from tests.server.gemini_doubles import TEMPLATE, Recording, loader
from tests.server.gemini_names import (
    ANSWERED,
    ARGUMENTS,
    ASKED,
    BASE,
    CACHED,
    CALLER,
    DESCRIPTION,
    DOCUMENT,
    ENTRIES,
    ENVELOPE,
    FLAKY,
    GUIDED,
    MODEL,
    MUTE,
    PREAMBLE,
    PROFILE,
    PROSE,
    RESULT,
    REUSED,
    SCHEMA,
    STRANGER,
    SYSTEM,
    TOOL_BODY,
    WRITER,
)

scanning = importlib.import_module("mlx_omnia.server.services.catalog.scan")
"""The module `scan()` reads its two cache paths out of. `catalog` re-exports them, and a
rebound re-export is not what the function looks at."""



@dataclass(frozen=True)
class Stand:
    base_url: str
    engine: Recording


@pytest.fixture(scope="module")
def fresh_state() -> None:
    """The conftest wipe, once for the module instead of once per test: the server here is a
    real one and outlives every test in the file, and a state directory removed under it is
    the profile row and the open database going with it."""
    root = state_dir()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def seed_profile() -> None:
    """The profile the name-with-a-colon tests select, written before the app boots — the same
    row `services.profiles.save` writes, as the row rather than through the async service."""
    with closing(sqlite3.connect(server_db())) as connection, connection:
        connection.execute(
            "INSERT INTO profiles(model, name, sampling, system_prompt, template, features)"
            " VALUES(?, ?, ?, ?, NULL, '{}')",
            (
                MODEL,
                PROFILE,
                Sampling(temperature=0.0).model_dump_json(exclude_none=True),
                SYSTEM,
            ),
        )


@pytest.fixture(scope="module")
def stand(fresh_state: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    root = tmp_path_factory.mktemp("gemini")
    daemon = Daemon()
    engine = Recording(loader, daemon)
    app = app_of(engine, daemon)

    # The catalog reads the machine's real Hugging Face cache and `create_app` mounts it:
    # patched for the whole module so nothing here can touch what the user has downloaded.
    with pytest.MonkeyPatch.context() as patched:
        for module in (catalog, scanning):
            patched.setattr(module, "HUB_CACHE", root / "hub")
            patched.setattr(module, "QUANTIZED_CACHE", root / "quantized")
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        # After the lifespan, which is what creates the tables this row goes into.
        seed_profile()
        yield Stand(base_url=f"http://127.0.0.1:{port}", engine=engine)
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive(), "the stand's server did not shut down"


@pytest.fixture(scope="module")
def client(stand: Stand) -> genai.Client:
    """`vertexai=False` and an explicit key so the environment cannot decide: this SDK reads
    `GOOGLE_GENAI_USE_VERTEXAI` and two key variables when it is left to guess."""
    return genai.Client(
        api_key="unused",
        vertexai=False,
        http_options=types.HttpOptions(base_url=f"{stand.base_url}/api/gemini"),
    )


def candidate(chunk: types.GenerateContentResponse) -> types.Candidate:
    assert chunk.candidates is not None and len(chunk.candidates) == 1
    return chunk.candidates[0]


def url(stand: Stand, tail: str) -> str:
    return f"{stand.base_url}/api/gemini/v1beta/models/{tail}"


def offered(mode: types.FunctionCallingConfigMode | None = None) -> types.GenerateContentConfig:
    """The declaration through the SDK's own type. `parameters_json_schema` and not
    `parameters`: the second one goes through the SDK's `Schema`, which spells the types
    `OBJECT` and `STRING` — a schema of its own, where this one is the one the other two
    dialects send."""
    declaration = types.FunctionDeclaration(
        name="get_weather", description=DESCRIPTION, parameters_json_schema=SCHEMA
    )
    chosen = (
        None
        if mode is None
        else types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
    )
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])], tool_config=chosen
    )


def submitted(stand: Stand) -> Chat:
    """The conversation the engine was handed. It reaches no response body — the template
    renders from it and from nothing else, so a key the conversion drops is a key no checkpoint
    can put back."""
    job = stand.engine.jobs[-1]
    assert isinstance(job.input, Chat)
    return job.input


__all__ = [
    "ANSWERED",
    "ARGUMENTS",
    "ASKED",
    "BASE",
    "CACHED",
    "CALLER",
    "DESCRIPTION",
    "DOCUMENT",
    "ENTRIES",
    "ENVELOPE",
    "FLAKY",
    "GUIDED",
    "MODEL",
    "MUTE",
    "PREAMBLE",
    "PROFILE",
    "PROSE",
    "RESULT",
    "REUSED",
    "SCHEMA",
    "STRANGER",
    "SYSTEM",
    "TEMPLATE",
    "TOOL_BODY",
    "WRITER",
    "Stand",
    "Turn",
    "candidate",
    "client",
    "fresh_state",
    "offered",
    "stand",
    "submitted",
    "url",
]
