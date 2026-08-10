"""37.1: an error in the dialect of whoever asked, over the app `create_app` actually builds.

The three suites beside this one each mount one router on an app of their own, which is what
makes them tests of a route. What decides the envelope of a *refused body*, though, is not the
route — the request never reaches one — but the handler `create_app` registers and the prefix
it dispatches on. So the app here is the daemon's own, with every dialect on it at once, and
the judge is each SDK's typed exception rather than the status line.

The two ways an SDK treats an error body, and both hurt: OpenAI's and Anthropic's pick the
exception class by status and attach the body raw, so what `type` says inside it is what a
client's own error mapping reads; Gemini's *parses* the body, and an envelope without
`error.status`, `error.message` and `error.code` prints literally as `None None.` — which is
what a client is left holding when it gets somebody else's shape.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeIs

import anthropic
import httpx
import pytest
import uvicorn
from google import genai
from google.genai import errors, types
from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI

from sideros import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
)
from sideros.parsers import Segment
from sideros_server import Engine, catalog, create_app
from sideros_server.store import Store

MODEL = "stand/echo"
"""A `/` in the id like every Hub repository: the Gemini route matches a tail that carries
slashes, and the error tests below name a model that resolves."""

KEY = "sk-sideros-37-1"

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


@dataclass(frozen=True)
class Stand:
    base_url: str
    store: Store


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    root = tmp_path_factory.mktemp("dialect-errors")
    store = Store(root / "server.db")
    app = create_app(Engine(loader), store)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # The catalog reads the machine's real Hugging Face cache, and `create_app` mounts it:
    # patched for the whole module so nothing here can touch what the user has downloaded.
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(catalog, "HUB_CACHE", root / "hub")
        patched.setattr(catalog, "QUANTIZED_CACHE", root / "quantized")
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        yield Stand(base_url=f"http://127.0.0.1:{port}", store=store)
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


def envelope(body: object) -> dict[str, object]:
    """The error object, whichever of the two shapes the client handed over: the OpenAI SDK
    unwraps `{"error": {...}}` before it attaches the body to the exception and the Anthropic
    one does not, so a test that opened only one of them would be asserting about the SDK."""
    assert isinstance(body, dict), f"expected an object, got {body!r}"
    error = body.get("error", body)
    assert isinstance(error, dict), f"expected an error object, got {error!r}"
    return error


def text(value: object) -> str:
    assert isinstance(value, str), f"expected a string, got {value!r}"
    return value


def test_an_unknown_model_is_each_sdks_own_not_found(
    openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """The status is what the first two dispatch on, and the body is what the client's error
    mapping reads afterwards. The third reads the body for everything."""
    with pytest.raises(NotFoundError) as by_openai:
        openai.chat.completions.create(
            model="nope", messages=[{"role": "user", "content": "Hi"}], max_tokens=4
        )
    error = envelope(by_openai.value.body)
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "model_not_found"

    with pytest.raises(anthropic.NotFoundError) as by_claude:
        claude.messages.create(
            model="nope", messages=[{"role": "user", "content": "Hi"}], max_tokens=4
        )
    body = by_claude.value.body
    assert isinstance(body, dict) and body["type"] == "error"
    assert envelope(body)["type"] == "not_found_error"

    with pytest.raises(errors.ClientError) as by_gemini:
        gemini.models.generate_content(model="nope", contents="Hi")
    assert by_gemini.value.code == 404
    assert by_gemini.value.status == "NOT_FOUND"


def test_a_refused_body_comes_back_in_the_dialect_of_the_route(
    openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """The half no route decides: the request is refused before one is chosen, and what used to
    answer was FastAPI's `{"detail": [...]}` — nobody's dialect — and then OpenAI's for all
    three. The field is named in the message, because a client that cannot see which of its
    own fields we mean is left rewriting the request at random.

    `extra_body` is how a body the SDK's own types would not write reaches the wire: it is
    merged into the JSON after every known field, so `max_tokens` arrives as text.
    """
    with pytest.raises(BadRequestError) as by_openai:
        openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4,
            extra_body={"max_tokens": "lots"},
        )
    error = envelope(by_openai.value.body)
    assert error["code"] == "invalid_request"
    assert "max_tokens" in text(error["message"])

    with pytest.raises(anthropic.BadRequestError) as by_claude:
        claude.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4,
            extra_body={"max_tokens": "lots"},
        )
    body = by_claude.value.body
    assert isinstance(body, dict) and body["type"] == "error"
    error = envelope(body)
    assert error["type"] == "invalid_request_error"
    assert "max_tokens" in text(error["message"])

    # `candidateCount` is a real field of this dialect and not one this server honours, so the
    # SDK writes it and `extra="forbid"` refuses it — the same handler, reached through the
    # client's own types.
    with pytest.raises(errors.ClientError) as by_gemini:
        gemini.models.generate_content(
            model=MODEL, contents="Hi", config=types.GenerateContentConfig(candidate_count=2)
        )
    failure = by_gemini.value
    assert failure.code == 400
    assert failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "candidateCount" in failure.message


def test_the_gemini_refusal_prints_as_a_message_and_not_as_none_none(
    gemini: genai.Client,
) -> None:
    """`APIError.__str__` is built from what it parsed out of the body. With OpenAI's envelope
    on the wire it has none of the three keys and prints literally `None None.`, which tells a
    client nothing at all about what it got wrong — and that is the whole reason this dialect
    encodes its own errors."""
    with pytest.raises(errors.ClientError) as raised:
        gemini.models.generate_content(
            model=MODEL, contents="Hi", config=types.GenerateContentConfig(candidate_count=2)
        )

    printed = str(raised.value)
    assert "None None" not in printed
    assert "INVALID_ARGUMENT" in printed and "candidateCount" in printed


def test_a_malformed_body_under_a_prefix_with_no_route_still_gets_the_dialect(
    stand: Stand,
) -> None:
    """`/api/openai/v1/responses` is a second route group behind the same prefix, and the
    `/admin` group is behind none: what the handler dispatches on is the path, so both of them
    have to come out in OpenAI's shape rather than in FastAPI's."""
    refused = httpx.post(
        f"{stand.base_url}/api/openai/v1/responses",
        json={"model": MODEL, "input": "Hi", "max_output_tokens": "lots"},
        timeout=30,
    )
    assert refused.status_code == 400, refused.text
    assert "max_output_tokens" in text(envelope(refused.json())["message"])

    patched = httpx.patch(
        f"{stand.base_url}/admin/config", json={"port": "eight thousand"}, timeout=30
    )
    assert patched.status_code == 400, patched.text
    assert envelope(patched.json())["type"] == "invalid_request_error"


def test_a_wrong_api_key_is_refused_in_each_dialect(
    stand: Stand, openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """The middleware's own dispatch, which is the precedent this stage followed — asserted
    here over the real app, where the key is read from the store per request. Configured after
    the clients were built and cleared again at the end, so the rest of the module keeps
    talking to an open daemon."""
    stand.store.set_config({"api_key": json.dumps(KEY)})
    try:
        with pytest.raises(AuthenticationError) as by_openai:
            openai.models.list()
        assert envelope(by_openai.value.body)["code"] == "invalid_api_key"

        with pytest.raises(anthropic.AuthenticationError) as by_claude:
            claude.models.list()
        body = by_claude.value.body
        assert isinstance(body, dict) and body["type"] == "error"
        assert envelope(body)["type"] == "authentication_error"

        with pytest.raises(errors.ClientError) as by_gemini:
            gemini.models.generate_content(model=MODEL, contents="Hi")
        assert by_gemini.value.code == 401
        assert by_gemini.value.status == "UNAUTHENTICATED"
        assert "None None" not in str(by_gemini.value)
    finally:
        stand.store.set_config({"api_key": json.dumps(None)})

    assert [model.id for model in openai.models.list()] == []


def test_the_model_that_resolves_still_answers(openai: OpenAI) -> None:
    """The other half of every refusal above: the same app, a request it accepts, and a
    generation that comes back — a handler that swallowed good requests would leave all of
    them green."""
    answer = openai.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=4, temperature=0
    )

    assert answer.choices[0].message.content == "<user>Hi"


def test_the_stand_touches_no_hub_cache(stand: Stand) -> None:
    """`create_app` mounts the catalog, and the catalog reads the machine's own cache: what
    this module patched is what the listing answers from, and it is empty."""
    listed = httpx.get(f"{stand.base_url}/admin/models", timeout=30)

    assert listed.status_code == 200, listed.text
    assert listed.json() == []
