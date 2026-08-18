"""The stand the image suites share: one app with all four dialects on it, over a real port.

The model under the engine answers with the prompt it was given, an image spelled as its size
and a digest of its pixels. That is the only window a test has on this frontier: what the
dialect built is a `Chat`, what the capability made of it is a list of parts, and both live on
the other side of `stream`. Comparing the two `Chat`s directly is not open either — an `Image`
holds a numpy array, and dataclass equality over one is not a boolean.
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
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputParam

from mlx_omnia import (
    RGB_IMAGE,
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    Image,
    LanguageModel,
    LanguagePrompt,
    ModelInput,
    ModelSignature,
    MultimodalChatCapability,
    Text,
)
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.paths import state_dir
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Job, Loader
from mlx_omnia.server.services import catalog
from tests.server.conftest import app_of
from tests.server.png_fixtures import BASE64, DATA_URL, PIXELS, PNG, spelling

scanning = importlib.import_module("mlx_omnia.server.services.catalog.scan")
"""The module `scan()` reads its two cache paths out of. `catalog` re-exports them, and a
rebound re-export is not what the function looks at."""

MODEL = "stand/eyes"
TEXT_ONLY = "stand/deaf"
"""A checkpoint with a chat template and no vision tower: every conversation but one with an
image in it."""

BUDGET = 64

ASKED = "What is in this?"

MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
"""Qwen3.5's, which is what the capability cuts the render on."""

SOURCE = (
    "{% for message in messages %}<{{ message['role'] }}>"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for part in message['content'] %}"
    "{% if part.type == 'image' %}" + MARKER + "{% else %}{{ part.text }}{% endif %}"
    "{% endfor %}{% endif %}"
    "</{{ message['role'] }}>{% endfor %}"
)
"""One tag per turn, and the marker where an image is. Not a checkpoint's — what these tests
read is which parts reached the render and in which order. The two branches are the two shapes
`content` arrives in, which is what the real templates do as well (`content is iterable and
content is not mapping`)."""

TEMPLATE = ChatTemplate.from_source(SOURCE)

ANSWER = f"<user>{ASKED}{spelling(PIXELS)}</user>"
"""What every dialect has to come back with: the render, cut at the marker, with the image
between the two halves of it."""


@dataclass(frozen=True)
class Eyes:
    """Answers with the prompt it was handed — text as it stands, an image as its size and a
    digest of its pixels. A double that answered anything of its own would leave this whole
    frontier untested: what a dialect built is only visible in what reached the model."""

    vision: bool = True

    @property
    def native_signature(self) -> ModelSignature:
        inputs = frozenset({TEXT, RGB_IMAGE}) if self.vision else frozenset({TEXT})
        return ModelSignature(inputs, frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text | LanguagePrompt]:
        if isinstance(input, Text):
            return True
        return self.vision and isinstance(input, LanguagePrompt)

    def stream(self, input: Text | LanguagePrompt, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        parts = input.parts if isinstance(input, LanguagePrompt) else (input,)
        pieces: list[str] = []
        for part in parts:
            if isinstance(part, Text):
                pieces.append(part.read())
            else:
                assert isinstance(part, Image), f"unexpected part {part!r}"
                pieces.append(spelling(part.pixels))
        meter.prefill(sum(len(piece) for piece in pieces))
        for piece in pieces[: options.max_tokens]:
            meter.token()
            yield Segment("content", piece)


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id == MODEL:
        return CompositeModel(Eyes(), [MultimodalChatCapability(TEMPLATE, MARKER)])
    if model_id == TEXT_ONLY:
        return CompositeModel(Eyes(vision=False), [ChatCapability(TEMPLATE)])
    raise ValueError(f"no model {model_id!r} in this stand")


class Recording(Engine):
    """Keeps the jobs it hands out: what a turn of text became is a fact about the `Chat` and
    reaches no response body."""

    def __init__(self, loader: Loader, daemon: Daemon) -> None:
        super().__init__(loader, daemon, Metrics())
        self.jobs: list[Job] = []

    async def submit(
        self,
        model_id: str,
        input: ModelInput,
        options: GenerationOptions,
        reservation: object | None = None,
        batch_limit: int | None = None,
    ) -> Job:
        job = await super().submit(model_id, input, options, reservation, batch_limit)
        self.jobs.append(job)
        return job


@dataclass(frozen=True)
class Stand:
    base_url: str
    engine: Recording


@pytest.fixture(scope="module")
def fresh_state() -> None:
    """The conftest wipe, once for the module instead of once per test: the server here is a
    real one and outlives every test in the file, and a state directory removed under it takes
    the open database with it."""
    root = state_dir()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def stand(fresh_state: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    root = tmp_path_factory.mktemp("images")
    daemon = Daemon()
    engine = Recording(loader, daemon)
    app = app_of(engine, daemon)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # The catalog reads the machine's real Hugging Face cache and `create_app` mounts it:
    # patched for the whole module so nothing here can touch what the user has downloaded.
    with pytest.MonkeyPatch.context() as patched:
        # Both the package's names, which `create_app` watches, and the module the scan
        # itself reads.
        for module in (catalog, scanning):
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
        yield Stand(base_url=f"http://127.0.0.1:{port}", engine=engine)
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
    return genai.Client(
        api_key="unused",
        vertexai=False,
        http_options=types.HttpOptions(base_url=f"{stand.base_url}/api/gemini"),
    )


def through_chat(client: OpenAI, model: str = MODEL, url: str = DATA_URL) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ASKED},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    answer = client.chat.completions.create(
        model=model, messages=messages, max_tokens=BUDGET, temperature=0
    )
    content = answer.choices[0].message.content
    assert content is not None
    return content


def through_responses(client: OpenAI, model: str = MODEL, url: str = DATA_URL) -> str:
    given: ResponseInputParam = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": ASKED},
                {"type": "input_image", "image_url": url, "detail": "auto"},
            ],
        }
    ]
    answer = client.responses.create(
        model=model, input=given, max_output_tokens=BUDGET, temperature=0
    )
    return answer.output_text


def through_anthropic(client: anthropic.Anthropic, model: str = MODEL) -> str:
    reply = client.messages.create(
        model=model,
        max_tokens=BUDGET,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ASKED},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": BASE64,
                        },
                    },
                ],
            }
        ],
    )
    block = reply.content[0]
    assert block.type == "text", f"expected a text block, got {block.type!r}"
    return block.text


def through_gemini(client: genai.Client, model: str = MODEL, mime_type: str = "image/png") -> str:
    answer = client.models.generate_content(
        model=model,
        contents=[
            types.UserContent(
                parts=[
                    types.Part.from_text(text=ASKED),
                    types.Part.from_bytes(data=PNG, mime_type=mime_type),
                ]
            )
        ],
        config=types.GenerateContentConfig(max_output_tokens=BUDGET, temperature=0),
    )
    text = answer.text
    assert text is not None
    return text
