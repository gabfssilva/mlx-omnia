"""The app `create_app` builds, on a real socket, with every dialect's SDK pointed at it.

Shared by `test_dialect_errors.py`, which is where what it is a stand *for* is written down.
"""

import importlib
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeIs

import anthropic
import pytest
import uvicorn
from google import genai
from google.genai import types
from openai import OpenAI

from mlx_omnia import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
    paths,
)
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.services import catalog
from tests.server.conftest import wired

MODEL = "stand/echo"
"""A `/` in the id like every Hub repository: the Gemini route matches a tail that carries
slashes, and the error tests below name a model that resolves."""

KEY = "sk-mlx_omnia-37-1"

SOURCE = "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}{% endfor %}"

TEMPLATE = ChatTemplate.from_source(SOURCE)


@dataclass(frozen=True)
class Echo:
    """Answers with the prompt it was handed. Nothing here is about the generation — what is
    under test is what comes back when the request never becomes one — but a model that
    answers keeps the "this one works" half of every test honest."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value))
        meter.token()
        yield Segment("content", input.value)


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id != MODEL:
        raise ValueError(f"no model {model_id!r} in this stand")
    return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])


@pytest.fixture(autouse=True)
def fresh_state() -> None:
    """The harness wipes the state directory before every test; this module's stand is
    module-scoped and holds the database open, so the wipe happens once, with the stand."""


@dataclass(frozen=True)
class Stand:
    base_url: str


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    root = tmp_path_factory.mktemp("dialect-errors")
    shutil.rmtree(paths.state_dir(), ignore_errors=True)
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    app = wired(loader)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # The catalog reads the machine's real Hugging Face cache, and `create_app` mounts it:
    # patched for the whole module so nothing here can touch what the user has downloaded.
    # Both the package's names and the scanner's own — the package re-exports the paths as
    # values, and `scan()` reads its own module's, so patching one is patching half of it.
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
        yield Stand(base_url=f"http://127.0.0.1:{port}")
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive(), "the stand's server did not shut down"


@pytest.fixture(scope="module")
def openai(stand: Stand) -> OpenAI:
    return OpenAI(base_url=f"{stand.base_url}/api/openai/v1", api_key="unused", max_retries=0)


@pytest.fixture(scope="module")
def claude(stand: Stand) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=f"{stand.base_url}/api/anthropic", api_key="unused", max_retries=0, timeout=60
    )


@pytest.fixture(scope="module")
def gemini(stand: Stand) -> genai.Client:
    """`vertexai=False` and an explicit key so the environment cannot decide: this SDK reads
    `GOOGLE_GENAI_USE_VERTEXAI` and two key variables when it is left to guess."""
    return genai.Client(
        api_key="unused",
        vertexai=False,
        http_options=types.HttpOptions(base_url=f"{stand.base_url}/api/gemini"),
    )
