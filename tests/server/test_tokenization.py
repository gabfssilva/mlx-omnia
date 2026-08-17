"""POST /admin/models/{id}/tokenize: the ids of a resident model's own tokenizer.

Two things decide whether this route is right. The tokenizer has to be the one belonging to
the model named in the path, reached through the facades `load` wrapped it in — which is
why every double here is the production shape, a `CompositeModel` carrying a chat template
over the task-level model that actually holds the tokenizer. And a model that is not
resident has to be refused: the caller is a chat window counting its context as the user
types, and a 30B loading because someone typed is the failure this route can cause and
nothing else can.
"""

import asyncio
import threading
from collections.abc import Iterator, Sequence
from typing import TypeIs

import httpx
import mlx.core as mx
from fastapi import FastAPI

from mlx_omnia import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    KVCache,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
    Tokenizer,
)
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server import catalog
from mlx_omnia.server.engine import Engine
from mlx_omnia.server.tokenization import router

TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

SLASHED = "mlx-community/Qwen3-0.6B-4bit"


class ConstantLM:
    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return mx.zeros((1, ids.shape[1], 8), dtype=mx.float32)


class CharTokenizer:
    """One id per character, shifted by `offset`: two of these disagree on every text, which
    is what makes an answer traceable to the tokenizer that produced it."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([self.offset + ord(character) for character in whole])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(identifier - self.offset for identifier in ids)


class RecordingTokenizer(CharTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.thread: threading.Thread | None = None

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        self.thread = threading.current_thread()
        return iter(super().encode(whole))


class Tokenizerless:
    """A model holding no tokenizer at all — what a test double is, and the one thing the
    route cannot answer for."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def resident(tokenizer: Tokenizer) -> CompositeModel[Text, Segment, GenerationOptions]:
    """The chain a checkpoint with a chat template loads as: the composite the engine holds,
    the task-level model under it, and the tokenizer on that one."""
    return CompositeModel(
        TextLanguageModel(ConstantLM(), tokenizer),
        [ChatCapability(ChatTemplate.from_source(TEMPLATE))],
    )


def application(engine: Engine) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine
    app.include_router(router)
    return app


async def post(app: FastAPI, model_id: str, text: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tokenize") as client:
        return await asyncio.wait_for(
            client.post(f"/admin/models/{model_id}/tokenize", json={"text": text}), 30
        )


def ids_of(response: httpx.Response) -> list[int]:
    body = response.json()
    assert isinstance(body, dict)
    ids = body["ids"]
    assert isinstance(ids, list)
    return ids


def detail_of(response: httpx.Response) -> str:
    body = response.json()
    assert isinstance(body, dict)
    detail = body["detail"]
    assert isinstance(detail, str)
    return detail


def test_the_ids_are_the_ones_the_named_models_tokenizer_produces() -> None:
    """Two residents with two tokenizers, and the id in the path is what picks between them:
    a route answering with whatever tokenizer it found first would count a conversation
    against a vocabulary that is not the one about to generate it.

    Both are composites, which is the other half of it — a walk that stops at the outermost
    model finds no tokenizer on either.
    """
    tokenizers = {"first": CharTokenizer(), "second": CharTokenizer(offset=1000)}

    async def run() -> tuple[list[int], list[int]]:
        engine = Engine(lambda model_id: resident(tokenizers[model_id]))
        for model_id in tokenizers:
            await engine.resolve(model_id)
        app = application(engine)
        first = await post(app, "first", "hi")
        second = await post(app, "second", "hi")
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        return ids_of(first), ids_of(second)

    first, second = asyncio.run(run())
    assert first == [ord("h"), ord("i")]
    assert second == [1000 + ord("h"), 1000 + ord("i")]


def test_a_model_that_is_not_resident_is_refused_and_nothing_is_loaded() -> None:
    """Tokenizing must not be a way to load a 30B by accident: the count is asked for on
    every keystroke, and the load is asked for on purpose through the residency route."""
    loads: list[str] = []

    async def run() -> tuple[httpx.Response, list[str], httpx.Response]:
        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return resident(CharTokenizer())

        engine = Engine(loader)
        app = application(engine)
        refused = await post(app, "cold", "hi")
        # Read where the claim is about, not at the end: the resolve below loads on purpose,
        # so a list read after it says nothing about what the refused request did.
        asked = list(loads)
        # And it is residency that decided it, not the route failing for its own reasons.
        await engine.resolve("cold")
        return refused, asked, await post(app, "cold", "hi")

    refused, asked, accepted = asyncio.run(run())
    assert refused.status_code == 409, refused.text
    assert "not resident" in detail_of(refused)
    assert asked == [], "counting tokens started a load"
    assert accepted.status_code == 200, accepted.text
    assert loads == ["cold"]


def test_an_id_with_a_slash_reaches_the_route_and_not_the_catalogs_model_path() -> None:
    """Model ids carry `/`, and the catalog's `{model_id:path}` matches slashes: without the
    same converter here, `mlx-community/x/tokenize` is a path this route never sees."""

    async def run() -> httpx.Response:
        engine = Engine(lambda _: resident(CharTokenizer()))
        await engine.resolve(SLASHED)
        app = application(engine)
        # The order `create_app` has to register them in, which is the one this asserts.
        app.include_router(catalog.router)
        return await post(app, SLASHED, "hi")

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    assert ids_of(response) == [ord("h"), ord("i")]


def test_the_encode_does_not_run_on_the_event_loop() -> None:
    """The BPE is Python and a full context is hundreds of thousands of characters. The loop
    this would sit on is the one handing SSE frames to whatever is generating."""
    tokenizer = RecordingTokenizer()

    async def run() -> httpx.Response:
        engine = Engine(lambda _: resident(tokenizer))
        await engine.resolve("counted")
        return await post(application(engine), "counted", "hi")

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    assert tokenizer.thread is not None
    assert tokenizer.thread is not threading.main_thread(), "the encode ran on the loop"


def test_a_resident_model_that_holds_no_tokenizer_says_so() -> None:
    """Not reachable through `mlx_omnia.load` — every architecture builds one — but the engine
    takes any loader, and a `None` that reached `encode` would be a 500 with no name in it."""

    async def run() -> httpx.Response:
        engine = Engine(lambda _: CompositeModel(Tokenizerless(), []))
        await engine.resolve("mute")
        return await post(application(engine), "mute", "hi")

    response = asyncio.run(run())
    assert response.status_code == 500, response.text
    assert "mute" in detail_of(response)
