"""`/api/anthropic/v1/*` against a real server process, judged by the official SDK.

The model under the engine answers with the prompt it was handed, in fixed-size pieces. That
is the only window a test has on what this dialect exists to do: `system` arrives as a field
of the request and has to leave as a turn of the conversation, and what turns one into the
other happens inside the engine, behind a chat template. A double that answered with
something of its own would leave the translation untested — the answer here *is* the rendered
conversation.

Streaming is judged by `client.messages.stream`, not by our reading of the frames: the SDK's
decoder dispatches on the `event:` line and drops a frame that carries none, and its
accumulator raises on an event that arrives before `message_start`. One raw-frame test sits
beside it to pin the sequence itself, which the accumulator tolerates more of than the
dialect allows.

The server is a real one, like the metrics suite: `TestClient` and `ASGITransport` run the
whole response before handing it over, and both the SDK's stream and the ping below are about
frames arriving while the generation is still going.
"""

import json
import math
import socket
import threading
import time
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

import anthropic
import httpx
import mlx.core as mx
import pytest
import uvicorn
from anthropic.types import Message as Reply
from anthropic.types import MessageParam, OutputConfigParam, TextBlockParam, ToolParam
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from sideros import (
    TEXT,
    Chat,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    ModelInput,
    ModelSignature,
    Text,
    greedy,
)
from sideros.generate import Constraint
from sideros.suppress import Segment, Segmenter
from sideros_server import anthropic as dialect
from sideros_server import catalog
from sideros_server.app import _invalid_request
from sideros_server.engine import Engine, Job, Loader
from sideros_server.profiles import Sampling
from sideros_server.responses import ToolTurn
from sideros_server.store import Profile, Store

ECHO = "echo"
FLAKY = "flaky"
SLOW = "slow"
BASE = "base"
"""A model with no chat template: the conversation has nowhere to go."""

CALLER = "caller"
MUTE = "mute"
STRANGER = "stranger"
"""Scripted callers: what a checkpoint writes when it is offered a function is the
checkpoint's own decision, so the generation is pinned and everything around it — the
template, the segmentation, the frames — stays real. `MUTE` writes the call and nothing
else, which is the answer that has to arrive with no text block at all; `STRANGER` writes the
same call behind a template that spells no envelope this server parses."""

GUIDED = "guided"
"""The one model here a grammar can be built over, by fiat in `Recording` below: nothing under
a stand of doubles holds a token table, and what a request carrying `output_config` has to
prove is that the route compiles the schema and hands the walk to the generation. It echoes
like `ECHO` does, so the same answer says which turns reached the prompt."""

COUNTED = "counted"
"""The one model here that holds a tokenizer, which is what `count_tokens` answers with: one
id per character (`Letters`), so the count a test expects is the length of the prompt the
template wrote and not a number read back off the thing under test."""

CATALOGUED = "vendor/tiny"
"""The one entry the fake hub cache holds, so the listing test is about what this file wrote
and not about what the machine happens to have downloaded."""

PRESET = "You are terse."
"""The profile's system prompt, which no request in this file ever sends."""

BUDGET = 999
"""Larger than any answer here, so a test that is not about the budget never reaches it."""

CHUNK = 8
"""Characters per piece the echo hands out, and therefore per meter mark: the answer arrives
in several frames, which is what gives the SDK's accumulator something to accumulate."""

SOURCE = (
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>{% endif %}"
    "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}"
    "{% for call in message.tool_calls %}"
    "<call>{{ call.function.name }}{{ call.function.arguments }}</call>"
    "{% endfor %}</{{ message['role'] }}>{% endfor %}"
)
"""One tag per turn with the role in it, and the tools and calls beside them. Nothing a real
checkpoint ships, and that is the point: what these tests read is which turns reached the
render, and in which order.

A call is spelled `<call>`, which is no family's marker, so `tool_family_of` says nothing about
this template and the tool channel stays shut. That is what the echo needs: it answers with the
prompt it was handed, and a prompt describing a call must not come back read as one.
"""

CALLING = SOURCE.replace("<call>", "<tool_call>").replace("</call>", "</tool_call>")
"""The same template spelling a call the way Qwen does, which is what a checkpoint's own
template does and what says which family it speaks. The callers below are loaded with this one
and the echo is not, and that is the whole difference between them."""

TEMPLATE = ChatTemplate.from_source(SOURCE)
CALLING_TEMPLATE = ChatTemplate.from_source(CALLING)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

DESCRIPTION = "Current weather in a city"

TOOLS: list[ToolParam] = [
    {"name": "get_weather", "description": DESCRIPTION, "input_schema": SCHEMA}
]

ENTRIES: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {"name": "get_weather", "description": DESCRIPTION, "parameters": SCHEMA},
    },
)
"""The same function in `chat/completions`'s nested shape, which is what every template reads
and what this dialect's `input_schema` has to become. Spelled out rather than converted: a
test that asked the route for the shape would agree with whatever the route did."""

CALL_PIECES = (
    "Let me check.",
    "<tool",
    '_call>\n{"name": "get_weather", "arguments": ',
    '{"city": "Paris"}}\n</tool',
    "_call>",
)
"""One generation, cut where a detokenizer would not: both markers straddle two pieces, so a
route that hands a piece out before it can tell hands out half a marker."""

PREAMBLE = CALL_PIECES[0]
ENVELOPE = "".join(CALL_PIECES[1:])
ARGUMENTS = '{"city": "Paris"}'
RESULT = "22 C"
ANSWERED = f"It is {RESULT}."

STALL = dialect._KEEP_ALIVE_SECONDS * 3
"""Three keep-alives before the first token: the slow model owes the stream a ping long
before it owes it anything else."""


def rendered(*turns: tuple[str, str]) -> str:
    """What the template writes for those turns, spelled out here rather than rendered: a
    test that asked the template what to expect would agree with itself whatever the dialect
    handed it."""
    return "".join(f"<{role}>{content}</{role}>" for role, content in turns)


@dataclass(frozen=True)
class Echo:
    """Answers with the prompt it was given, cut into pieces of `CHUNK` characters.

    It fills the meter the way `stream_ids` does — the prompt before the first piece, one
    mark per piece — because `usage` and `stop_reason` are read off it: a double that left
    the meter at zero would let both assertions pass on any number at all.
    """

    stall: float = 0.0
    fails: bool = False

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        if self.stall:
            time.sleep(self.stall)
        meter.prefill(len(input.value))
        pieces = [input.value[at : at + CHUNK] for at in range(0, len(input.value), CHUNK)]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(input.tool_family, prompt=input.value)
        for piece in pieces[: options.max_tokens]:
            meter.token()
            yield from segmenter.push(piece)
        yield from segmenter.flush()
        if self.fails:
            raise RuntimeError("the decode thread gave out")


@dataclass(frozen=True)
class Caller:
    """Answers with a call, in the pieces it was given — and with the result of that call once
    the conversation carries one, which is what tells the second turn of a round trip from the
    first: the result only reaches here if the `tool_result` block became a turn the template
    rendered."""

    pieces: tuple[str, ...]

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value))
        result = input.value.partition("<tool>")[2].partition("</tool>")[0]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(input.tool_family, prompt=input.value)
        for piece in (ANSWERED,) if result == RESULT else self.pieces:
            meter.token()
            yield from segmenter.push(piece)

        yield from segmenter.flush()


@dataclass(frozen=True)
class Letters:
    """One id per character. `count_tokens` is about which text was rendered and encoded, not
    about a vocabulary: with this, the count a test asserts is a length it can write down."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)


@dataclass(frozen=True)
class Counted(Echo):
    """The echo, with a tokenizer under it. It is a model of its own rather than a field on
    `Echo` because `tokenizer_of` walks down to whatever holds one: giving every double a
    tokenizer would change what the grammar refusal below is about."""

    tokenizer: Letters = Letters()


def load(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
    match model_id:
        case "echo":
            return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])
        case "counted":
            return CompositeModel(Counted(), [ChatCapability(TEMPLATE)])
        case "flaky":
            return CompositeModel(Echo(fails=True), [ChatCapability(TEMPLATE)])
        case "slow":
            return CompositeModel(Echo(stall=STALL), [ChatCapability(TEMPLATE)])
        case "base":
            return CompositeModel(Echo(), [])
        case "caller":
            return CompositeModel(Caller(CALL_PIECES), [ChatCapability(CALLING_TEMPLATE)])
        case "mute":
            return CompositeModel(Caller(CALL_PIECES[1:]), [ChatCapability(CALLING_TEMPLATE)])
        case "stranger":
            return CompositeModel(Caller(CALL_PIECES), [ChatCapability(TEMPLATE)])
        case "guided":
            return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])
        case other:
            raise ValueError(f"no model {other!r} in this stand")


class Free:
    """A walk that forbids nothing.

    No model on this stand holds a token table, so a real `Vocabulary` cannot be built over
    one and every schema would end in the same refusal. What a request carrying a schema has
    to prove here is the wiring — that the route compiles it and hands the walk to the
    generation — and that is what this stands in for.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return logits

    def accept(self, token: int) -> bool:
        return True


class Recording(Engine):
    """Keeps the jobs it hands out and the schemas it was asked to compile.

    Only `GUIDED` gets the double above: every other id falls through to the engine's own
    `constrain`, which is what makes the refusal below a real one — nothing under a double has
    a tokenizer, a head width or a stop id to compile against.
    """

    def __init__(self, loader: Loader) -> None:
        super().__init__(loader)
        self.jobs: list[Job] = []
        self.compiled: list[Mapping[str, object]] = []

    async def submit(self, model_id: str, input: ModelInput, options: GenerationOptions) -> Job:
        job = await super().submit(model_id, input, options)
        self.jobs.append(job)
        return job

    async def constrain(self, model_id: str, schema: Mapping[str, object]) -> Constraint:
        if model_id != GUIDED:
            return await super().constrain(model_id, schema)
        self.compiled.append(schema)
        return Free()


def fake_hub(root: Path) -> None:
    """One checkpoint in the shape `catalog.scan` recognizes: a snapshot with a config and
    the weights it promises. `refs/main` is left out — the scan falls back to the newest
    revision, and there is only one."""
    snapshot = root / f"models--{CATALOGUED.replace('/', '--')}" / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "tiny"}), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"")


@dataclass(frozen=True)
class Stand:
    base_url: str
    url: str
    """`/api/anthropic/v1/messages`, which the raw-frame tests post to directly."""
    engine: Recording


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    root = tmp_path_factory.mktemp("anthropic")
    fake_hub(root / "hub")
    store = Store(root / "server.db")
    empty = Sampling().model_dump_json(exclude_none=True)
    store.save_profile(Profile(model=ECHO, name="terse", sampling=empty, system_prompt=PRESET))
    store.save_profile(Profile(model=CATALOGUED, name="code", sampling=empty))
    engine = Recording(load)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.include_router(dialect.router)
    # The handler `create_app` registers, mounted here by hand: it picks the encoder by route
    # prefix, and every path on this app carries this dialect's. Without it a refused body
    # comes back as FastAPI's own 422 and the SDK raises the wrong class for the wrong reason.
    # That `create_app` registers it at all is `test_dialect_errors.py`'s.
    app.add_exception_handler(RequestValidationError, _invalid_request)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # The catalog reads the machine's real Hugging Face cache: patched for the whole module,
    # so the listing test answers about this fixture and touches nothing of the user's.
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


def test_the_system_field_becomes_the_first_turn_and_nothing_else_does(
    client: anthropic.Anthropic,
) -> None:
    """The translation this dialect exists for. The answer is the rendered conversation, so a
    `system` that was dropped, given the wrong role, or appended after the user's own turn is
    visible in it — and the request that sends none is what says the template is not writing
    a system turn by itself."""
    plain = ask(client, "Hello")
    assert only_text(plain) == rendered(("user", "Hello"))

    named = ask(client, "Hello", system="Answer in one word.")
    assert only_text(named) == rendered(("system", "Answer in one word."), ("user", "Hello"))

    blocks = ask(client, "Hello", system=[{"type": "text", "text": "Answer in one word."}])
    assert only_text(blocks) == only_text(named)


def test_a_profile_fills_the_system_the_request_left_out(client: anthropic.Anthropic) -> None:
    """`model:profile` is how a dialect with no field for a preset selects one. Its prompt
    becomes the conversation's first turn, and it loses to a request that sent one of its
    own — the same precedence the sampling knobs follow."""
    preset = ask(client, "Hello", model=f"{ECHO}:terse")
    assert only_text(preset) == rendered(("system", PRESET), ("user", "Hello"))

    own = ask(client, "Hello", model=f"{ECHO}:terse", system="Speak plainly.")
    assert only_text(own) == rendered(("system", "Speak plainly."), ("user", "Hello"))


def test_usage_counts_the_rendered_prompt_and_the_pieces_emitted(
    client: anthropic.Anthropic,
) -> None:
    """`input_tokens` is not the message the client sent: what reaches the model is the
    conversation the template rendered, which exists nowhere but inside the engine. The echo
    counts one prompt token per character of it, and one output token per piece."""
    prompt = rendered(("user", "Hello"))
    reply = ask(client, "Hello")

    assert reply.usage.input_tokens == len(prompt)
    assert reply.usage.output_tokens == math.ceil(len(prompt) / CHUNK)
    assert reply.role == "assistant"
    assert reply.type == "message"
    assert reply.model == ECHO


def test_the_budget_and_the_end_of_the_turn_are_two_stop_reasons(
    client: anthropic.Anthropic,
) -> None:
    """This dialect's vocabulary and not OpenAI's: `end_turn` where the other says `stop`.
    The two are told apart by the count, so the cut answer has to be short *and* say so."""
    whole = ask(client, "Hello")
    assert whole.stop_reason == "end_turn"
    assert whole.stop_sequence is None

    cut = ask(client, "Hello", max_tokens=2)
    assert cut.stop_reason == "max_tokens"
    assert cut.usage.output_tokens == 2
    assert only_text(cut) == rendered(("user", "Hello"))[: 2 * CHUNK]


def test_the_official_sdk_accumulates_the_named_events(client: anthropic.Anthropic) -> None:
    """What judges the frames is the SDK's own accumulator: an event it cannot name raises
    "unexpected event order" instead of folding in, and `input_tokens` reaches it through
    `message_start` alone — a stream that opened with a placeholder count would answer zero
    here while the non-streaming path stayed green."""
    prompt = rendered(("user", "Hello"))
    with client.messages.stream(model=ECHO, messages=turns("Hello"), max_tokens=BUDGET) as stream:
        deltas = list(stream.text_stream)
        final = stream.get_final_message()

    assert len(deltas) > 1, "the whole answer came in one frame; nothing was accumulated"
    assert "".join(deltas) == prompt
    assert only_text(final) == prompt
    assert final.role == "assistant"
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == len(prompt)
    assert final.usage.output_tokens == math.ceil(len(prompt) / CHUNK)


def test_the_stream_is_the_dialects_own_sequence_of_named_events(stand: Stand) -> None:
    """The shape the accumulator is more tolerant of than the dialect is: it would take a
    block that never closed, or a `message_stop` that never came. Every frame is named, and
    the name is also the `type` inside it — a mismatch is what makes an SDK construct the
    wrong member of the union."""
    captured = [(name, payload) for name, payload in frames(stand) if name != "ping"]
    names = [name for name, _ in captured]

    assert names[0] == "message_start"
    assert names[1] == "content_block_start"
    assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert set(names[2:-3]) == {"content_block_delta"}
    assert all(payload["type"] == name for name, payload in captured)

    opened = entry(captured[0][1]["message"])
    assert opened["content"] == [] and opened["stop_reason"] is None
    assert entry(opened["usage"])["input_tokens"] == len(rendered(("user", "Hello")))

    pieces = [text(entry(payload["delta"])["text"]) for _, payload in captured[2:-3]]
    assert "".join(pieces) == rendered(("user", "Hello"))
    assert entry(captured[-2][1]["delta"])["stop_reason"] == "end_turn"


def test_a_long_prefill_is_held_open_by_the_dialects_ping(stand: Stand) -> None:
    """`message_start` waits for the first piece, because that is when the prompt has been
    counted. Without the ping the connection would then go silent for the whole prefill,
    which is how a client loses it on a large prompt."""
    names = [name for name, _ in frames(stand, model=SLOW)]

    assert names[0] == "ping", "the stream said nothing until the model did"
    assert "message_start" in names


def test_an_unknown_model_is_the_dialects_own_not_found(client: anthropic.Anthropic) -> None:
    """The SDK picks the class by status; the body is what a client's error mapping reads,
    and it is this dialect's envelope rather than OpenAI's."""
    with pytest.raises(anthropic.NotFoundError) as raised:
        ask(client, "Hello", model="nope")

    kind, message = envelope(raised.value.body)
    assert kind == "not_found_error"
    assert "nope" in message


def test_a_model_without_a_chat_template_is_refused(client: anthropic.Anthropic) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go: a client
    error, not a 500 out of the worker."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        ask(client, "Hello", model=BASE)

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "chat template" in message


def test_a_field_the_dialect_cannot_honour_is_refused_by_name(
    stand: Stand, client: anthropic.Anthropic
) -> None:
    """`container` names a sandbox that runs somewhere else, and a client told it was honoured
    was told the answer came from a machine it did not. `max_tokens` is the other half —
    required here, so a body without one is refused instead of given a default this dialect
    never had. Both come back named, in the envelope."""
    response = httpx.post(
        stand.url,
        json={"model": ECHO, "messages": turns("Hello"), "max_tokens": 8, "container": "cont_1"},
        timeout=30,
    )
    assert response.status_code == 400
    kind, message = envelope(response.json())
    assert kind == "invalid_request_error"
    assert "container" in message

    response = httpx.post(stand.url, json={"model": ECHO, "messages": turns("Hello")}, timeout=30)
    assert response.status_code == 400
    kind, message = envelope(response.json())
    assert kind == "invalid_request_error"
    assert "max_tokens" in message


def test_a_generation_that_fails_is_told_in_both_shapes(
    stand: Stand, client: anthropic.Anthropic
) -> None:
    """Before the first frame the status is still ours; after it, it is 200 for ever and the
    only place left to say so is the dialect's own `error` event. A stream that ended in
    `message_stop` anyway would hand the client a truncated answer as a complete one."""
    with pytest.raises(anthropic.InternalServerError) as raised:
        ask(client, "Hello", model=FLAKY)
    kind, message = envelope(raised.value.body)
    assert kind == "api_error"
    assert "gave out" in message

    captured = frames(stand, model=FLAKY)
    names = [name for name, _ in captured]
    assert names[-1] == "error"
    assert "message_stop" not in names
    assert entry(captured[-1][1]["error"])["type"] == "api_error"

    with (
        pytest.raises(anthropic.APIStatusError) as broke,
        client.messages.stream(model=FLAKY, messages=turns("Hello"), max_tokens=BUDGET) as stream,
    ):
        stream.get_final_message()
    assert "gave out" in str(broke.value)


def test_the_models_route_lists_the_catalog_and_its_profiles(client: anthropic.Anthropic) -> None:
    """The catalog, not the residents — no dialect's schema carries the notion, and a list of
    loaded models is empty exactly at boot, which is when it is read. A model with a profile
    answers to two names and both are listed: a name a client cannot see is a preset it
    cannot select."""
    listed = list(client.models.list())

    assert [model.id for model in listed] == [CATALOGUED, f"{CATALOGUED}:code"]
    assert {model.type for model in listed} == {"model"}


ASKED = "Weather in Paris?"


def conversation(turns: tuple[ToolTurn, ...]) -> str:
    """What the template writes for the conversation `chat/completions` would have built out of
    the same round: the tools nested under `function`, the call on the assistant's turn, the
    result as a turn of its own. Rendered rather than spelled out, because what is under test
    is the conversation and not the toy template — the two sides are the same instrument, and
    what has to agree is what reached it."""
    return TEMPLATE.render(Chat(turns, tools=ENTRIES))


def only_use(reply: Reply) -> tuple[str, str]:
    """The one `tool_use` block of an answer: its id and the name it called. The id is the
    dialect's own (`toolu_`), because it is what a `tool_result` comes back addressed to."""
    uses = [block for block in reply.content if block.type == "tool_use"]
    assert len(uses) == 1, f"expected one call, got {reply.content!r}"
    use = uses[0]
    assert use.id.startswith("toolu_")
    assert use.input == {"city": "Paris"}
    return use.id, use.name


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(
    client: anthropic.Anthropic,
) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function and
    answers with a `tool_use` block beside its text, the result goes back as the `tool_result`
    block of the next user message, and the second answer is one only a model that was handed
    the result can give.

    `stop_reason` is this dialect's own vocabulary for it — `tool_use`, where OpenAI says
    `tool_calls` — and a client that reads anything else there never executes the call.
    """
    first = client.messages.create(
        model=CALLER, messages=[{"role": "user", "content": ASKED}], max_tokens=BUDGET, tools=TOOLS
    )

    assert [block.type for block in first.content] == ["text", "tool_use"]
    assert first.stop_reason == "tool_use"
    use_id, name = only_use(first)
    assert name == "get_weather"

    replay: list[MessageParam] = [
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": PREAMBLE},
                {
                    "type": "tool_use",
                    "id": use_id,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": use_id, "content": RESULT}],
        },
    ]
    second = client.messages.create(model=CALLER, messages=replay, max_tokens=BUDGET, tools=TOOLS)

    assert only_text(second) == ANSWERED
    assert second.stop_reason == "end_turn"


def test_the_tools_and_the_blocks_of_a_round_become_the_conversation_openai_would_build(
    client: anthropic.Anthropic,
) -> None:
    """The frontier this stage owns, read where it is visible: the echo answers with the
    rendered prompt, so what the blocks became is in the reply. A `tool_result` is a turn of
    its own and comes out before the text of the message that carried it; a `tool_use` is a key
    of the turn that made it; `input` becomes JSON text, because that is what the templates in
    circulation render.

    The comparison is against the conversation `chat/completions` builds out of the same round.
    Same characters into the model through two dialects, which is what makes one checkpoint
    answer both the same way.
    """
    reply = client.messages.create(
        model=ECHO,
        max_tokens=BUDGET,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": ASKED},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": PREAMBLE},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": RESULT},
                    {"type": "text", "text": "And Lyon?"},
                ],
            },
        ],
    )

    expected: tuple[ToolTurn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": PREAMBLE,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ARGUMENTS},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "toolu_1"},
        {"role": "user", "content": "And Lyon?"},
    )
    assert only_text(reply) == conversation(expected)


def test_the_sdk_accumulates_the_call_out_of_the_named_events(
    client: anthropic.Anthropic, stand: Stand
) -> None:
    """What judges the frames is the SDK's own accumulator: it indexes every delta into the
    block `content_block_start` announced, and reparses the tool input from the `partial_json`
    it has accumulated so far. A block index off by one lands the arguments in the text block.

    One delta with the arguments whole inside it is A11's decision, and the assertion the
    accumulator cannot make: it would concatenate any number of fragments into the same
    string.
    """
    with client.messages.stream(
        model=CALLER, messages=[{"role": "user", "content": ASKED}], max_tokens=BUDGET, tools=TOOLS
    ) as stream:
        deltas = list(stream.text_stream)
        final = stream.get_final_message()

    assert "".join(deltas) == PREAMBLE, "the envelope reached the client as text"
    assert [block.type for block in final.content] == ["text", "tool_use"]
    assert final.stop_reason == "tool_use"
    assert only_use(final)[1] == "get_weather"

    captured = frames(stand, model=CALLER, tools=TOOLS)
    fragments = [
        text(entry(payload["delta"])["partial_json"])
        for name, payload in captured
        if name == "content_block_delta" and "partial_json" in entry(payload["delta"])
    ]
    assert fragments == [ARGUMENTS], "the arguments arrived in fragments"
    opened = [payload for name, payload in captured if name == "content_block_start"]
    assert [entry(payload["content_block"])["type"] for payload in opened] == ["text", "tool_use"]
    assert [payload["index"] for payload in opened] == [0, 1]


def test_an_answer_that_only_called_something_has_no_text_block(
    client: anthropic.Anthropic, stand: Stand
) -> None:
    """The empty text block this dialect does not write: a block announced with nothing in it
    is an assistant that answered `""` before it called, and a client rendering the transcript
    shows a blank turn. The stream says the same thing — the call's block is index 0, which is
    what says the text block is opened on the first text there is and not before."""
    reply = client.messages.create(
        model=MUTE, messages=[{"role": "user", "content": ASKED}], max_tokens=BUDGET, tools=TOOLS
    )

    assert [block.type for block in reply.content] == ["tool_use"]
    assert reply.stop_reason == "tool_use"

    opened = [
        payload
        for name, payload in frames(stand, model=MUTE, tools=TOOLS)
        if name == "content_block_start"
    ]
    assert len(opened) == 1
    assert entry(opened[0]["content_block"])["type"] == "tool_use"
    assert opened[0]["index"] == 0


def test_tool_choice_none_neither_offers_the_tools_nor_reads_a_call_back(
    client: anthropic.Anthropic,
) -> None:
    """`none` is honoured where it can be honoured: the tools never reach the prompt, so there
    is nothing to call rather than an instruction not to. And a turn nobody was offered a tool
    for is text — answering with a call would be answering with the one thing the client asked
    us not to do."""
    reply = client.messages.create(
        model=MUTE,
        messages=[{"role": "user", "content": ASKED}],
        max_tokens=BUDGET,
        tools=TOOLS,
        tool_choice={"type": "none"},
    )

    assert only_text(reply) == ENVELOPE
    assert reply.stop_reason == "end_turn"


def test_a_checkpoint_whose_envelope_nothing_here_parses_answers_with_the_text(
    client: anthropic.Anthropic,
) -> None:
    """Which family a checkpoint speaks is a fact of its chat template and not of the text it
    writes: read off the output instead, Qwen3.6's `<tool_call><function=…>` is taken for
    Qwen's JSON and the envelope is held for a parser that cannot read it. A template that
    spells none leaves the channel shut, and what the model wrote reaches the client whole —
    text, and no call it never made."""
    reply = client.messages.create(
        model=STRANGER,
        messages=[{"role": "user", "content": ASKED}],
        max_tokens=BUDGET,
        tools=TOOLS,
    )

    assert only_text(reply) == PREAMBLE + ENVELOPE
    assert reply.stop_reason == "end_turn"


def test_forcing_a_call_is_refused_by_name(client: anthropic.Anthropic) -> None:
    """`any` and `tool` are a constraint on decoding and there is none here: answering `auto`
    to a client that asked for one is a call the model may never have made. Named in the
    message, in this dialect's envelope."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=ECHO,
            messages=[{"role": "user", "content": ASKED}],
            max_tokens=BUDGET,
            tools=TOOLS,
            tool_choice={"type": "any"},
        )

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "tool_choice" in message


def body(**fields: object) -> dialect.MessagesRequest:
    asked: dict[str, object] = {
        "model": ECHO,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
    }
    return dialect.MessagesRequest.model_validate(asked | fields)


def test_a_profile_fills_the_sampling_knobs_the_request_left_out() -> None:
    """Which knobs the request left out is `model_fields_set` and not their values: the
    dialect's default temperature is 1.0, so a profile read off the value would override a
    client that asked for 1.0 explicitly, and a profile setting 0.0 would read as unset. The
    knobs with no field in this dialect can only ever come from here.

    Off HTTP because the only observable difference is the sampler the engine is handed, and
    `greedy` is the one that can be compared by identity.
    """
    plain = dialect._options(body(), Sampling(), None)
    assert plain.max_tokens == 16
    assert plain.sampler is not greedy, "the dialect's default is drawn, not argmaxed"
    assert plain.penalty is None

    preset = dialect._options(body(), Sampling(temperature=0.0, repetition_penalty=1.5), None)
    assert preset.sampler is greedy
    assert preset.penalty is not None

    asked = dialect._options(body(temperature=1.0), Sampling(temperature=0.0), None)
    assert asked.sampler is not greedy, "an explicit temperature lost to the profile's"


SCHEMA_FORMAT: OutputConfigParam = {"format": {"type": "json_schema", "schema": SCHEMA}}
"""The one spelling this dialect has for structured output, typed as the SDK types it. There
is no flag to soften it: upstream the answer is decoded under the schema, not checked against
it afterwards, so this route compiles it into a grammar or refuses the request."""


def test_an_output_format_is_a_guarantee_and_reaches_the_generation_as_a_walk(
    stand: Stand, client: anthropic.Anthropic
) -> None:
    """The schema is compiled against the model the request named and the walk that comes back
    is what the generation runs under. A route that compiled it and dropped the walk would
    answer 200 with a free decode, which is the guarantee broken silently.

    Nothing goes into the prompt, and the echo is what says so: the mask is the whole of it,
    and a schema in the prompt as well would be paying for the answer twice. Nothing is checked
    afterwards either — an answer decoding could not make invalid is not measured again."""
    reply = client.messages.create(
        model=GUIDED,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=BUDGET,
        output_config=SCHEMA_FORMAT,
    )

    assert only_text(reply) == rendered(("user", "Hello"))
    assert stand.engine.compiled[-1] == SCHEMA
    assert stand.engine.jobs[-1].options.constraint is not None


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    client: anthropic.Anthropic,
) -> None:
    """A model of this stand is in no catalog and holds no tokenizer, so there is no token
    table to compile against. What the client must not get is the answer anyway: this field
    asks for a guarantee, and a 200 carrying a free decode is that guarantee broken silently.
    The way out is the client's to choose, which is why the reason is in the message."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=ECHO,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=BUDGET,
            output_config=SCHEMA_FORMAT,
        )

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "token table" in message


def test_an_output_format_and_tools_are_refused_together(client: anthropic.Anthropic) -> None:
    """The one combination the mask makes impossible rather than expensive: it allows the
    schema's ids from the first token, so an offered function can never be called, and a 200
    with the tools silently uncallable is the answer this refusal exists instead of.

    `tool_choice: {"type": "none"}` is not this case — the tools never enter the prompt, so
    there is nothing being dropped, and the request goes through."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=GUIDED,
            messages=[{"role": "user", "content": ASKED}],
            max_tokens=BUDGET,
            tools=TOOLS,
            output_config=SCHEMA_FORMAT,
        )

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "output_config.format" in message

    answered = client.messages.create(
        model=GUIDED,
        messages=[{"role": "user", "content": ASKED}],
        max_tokens=BUDGET,
        tools=TOOLS,
        tool_choice={"type": "none"},
        output_config=SCHEMA_FORMAT,
    )
    assert only_text(answered) == rendered(("user", ASKED))


def test_the_other_key_of_output_config_is_accepted_and_changes_nothing(
    client: anthropic.Anthropic,
) -> None:
    """`effort` and `metadata` ask for spend and accounting: there is no dial for the first
    under this server and nothing bills per user for the second, so both are accepted and
    neither reaches the prompt. Refusing them would leave a client that sends them on every
    request — Claude Code sends both — unable to talk to this dialect at all, and what they
    ask for is not an answer of another kind.

    The echo is what says they went nowhere: the answer is the rendered conversation, and it
    is the same one the request without them gets."""
    plain = ask(client, "Hello")
    with_dials = client.messages.create(
        model=ECHO,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=BUDGET,
        output_config={"effort": "low"},
        metadata={"user_id": "someone"},
    )

    assert only_text(with_dials) == only_text(plain) == rendered(("user", "Hello"))


def post(stand: Stand, **body: object) -> httpx.Response:
    """A body the SDK has no parameter for, sent as it is. Two fields here are beta upstream
    and typed nowhere in the client — `context_management` is the one Claude Code sends on
    every request — and a test that reached them through `extra_body` would be asserting
    about the SDK's escape hatch instead of about the route."""
    asked: dict[str, object] = {
        "model": ECHO,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": BUDGET,
    }
    return httpx.post(stand.url, json=asked | body, timeout=30)


def test_a_stop_sequence_cuts_the_answer_and_comes_back_named(
    client: anthropic.Anthropic,
) -> None:
    """The engine's `stop` is a set of token ids and cannot express a string, so the sequence
    is honoured over the text: what the client reads ends where its sequence began, and the
    reason the turn ended says which one it was. `end_turn` beside a cut answer is a client
    told the model finished on its own."""
    reply = client.messages.create(
        model=ECHO, messages=turns("Hello"), max_tokens=BUDGET, stop_sequences=["llo"]
    )

    assert only_text(reply) == "<user>He"
    assert reply.stop_reason == "stop_sequence"
    assert reply.stop_sequence == "llo"

    ran_out = client.messages.create(
        model=ECHO, messages=turns("Hello"), max_tokens=BUDGET, stop_sequences=["nowhere"]
    )
    assert only_text(ran_out) == rendered(("user", "Hello"))
    assert ran_out.stop_reason == "end_turn"
    assert ran_out.stop_sequence is None


def test_a_sequence_split_across_two_pieces_is_still_one(stand: Stand) -> None:
    """`Hello` straddles the boundary between the first eight characters and the second, so
    a route matching piece by piece would never see it. What the client must also never see
    is the half that arrived first: the frames stop at `<user>`, and the `He` that could
    still have become the sequence is held until the piece that decides it does."""
    captured = frames(stand, stop_sequences=["Hello"])

    written = "".join(
        text(entry(payload["delta"])["text"])
        for named, payload in captured
        if named == "content_block_delta"
    )
    assert written == "<user>"
    ended = next(payload for named, payload in captured if named == "message_delta")
    assert entry(ended["delta"]) == {"stop_reason": "stop_sequence", "stop_sequence": "Hello"}


def test_a_system_turn_between_two_messages_is_rendered_where_it_sits(
    client: anthropic.Anthropic,
) -> None:
    """The other spelling of `system` in this dialect: the field opens the conversation and
    the role appends an instruction mid-way, which is what a client sends to steer a session
    without editing the prompt it has already cached. Both reach the template, and the order
    is the client's — an instruction hoisted to the front is a different conversation."""
    reply = client.messages.create(
        model=ECHO,
        max_tokens=BUDGET,
        system="You are terse.",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "system", "content": "Answer in French now."},
            {"role": "user", "content": "Again"},
        ],
    )

    assert only_text(reply) == rendered(
        ("system", "You are terse."),
        ("user", "Hello"),
        ("assistant", "Hi."),
        ("system", "Answer in French now."),
        ("user", "Again"),
    )


THINKS = "a<think>why</think>b"
"""What the echo is asked, and therefore what it answers: the reasoning marker is in the
prompt, so the segmenter inside the model cuts the answer into the two channels this dialect
has to tell apart. Nothing here scripts a channel by hand."""

THOUGHT = "<user>a"
"""The answer's text, first half — everything before the block. The second is `b</user>`."""


def test_reasoning_leaves_as_a_thinking_block_and_display_decides_its_text(
    client: anthropic.Anthropic,
) -> None:
    """Reasoning is a channel by the time it reaches this module, and this dialect has a block
    for it: text is what the model answered, and `<think>` drawn as the answer is the whole
    turn read wrong. `display` is about the client and not about the model — `omitted` leaves
    the block with its text empty, which is what a client that did not want to read the
    reasoning asked for, and not a turn that never reasoned.

    Three blocks and not two: the echo writes text, reasons, and writes again, and that is
    the order it comes back in. The stream can only hand blocks over as they arrive, so a
    message that gathered the reasoning into one block at the front would be this same
    generation read two different ways."""
    reply = ask(client, THINKS)

    assert [block.type for block in reply.content] == ["text", "thinking", "text"]
    opening, thinking, answer = reply.content
    assert opening.type == "text" and opening.text == THOUGHT
    assert thinking.type == "thinking" and thinking.thinking == "why"
    assert answer.type == "text" and answer.text == "b</user>"

    hidden = client.messages.create(
        model=ECHO,
        messages=turns(THINKS),
        max_tokens=BUDGET,
        thinking={"type": "adaptive", "display": "omitted"},
    )
    blocks = hidden.content
    assert [block.type for block in blocks] == ["text", "thinking", "text"]
    assert blocks[1].type == "thinking" and blocks[1].thinking == ""
    assert blocks[2].type == "text" and blocks[2].text == "b</user>"


def test_the_sdk_accumulates_the_thinking_block_out_of_the_named_events(
    client: anthropic.Anthropic,
) -> None:
    """The stream's side of the block above, judged by the SDK's own accumulator: a
    `thinking_delta` landing in a block that was opened as text is what it raises on, so the
    block turning with the channel is the whole of this."""
    with client.messages.stream(model=ECHO, messages=turns(THINKS), max_tokens=BUDGET) as stream:
        reply = stream.get_final_message()

    assert [block.type for block in reply.content] == ["text", "thinking", "text"]
    assert reply.content[0].type == "text" and reply.content[0].text == THOUGHT
    assert reply.content[1].type == "thinking" and reply.content[1].thinking == "why"
    assert reply.content[2].type == "text" and reply.content[2].text == "b</user>"


def test_a_thinking_block_replayed_by_the_client_is_read_and_left_out_of_the_prompt(
    client: anthropic.Anthropic,
) -> None:
    """The dialect asks a client to send the assistant's blocks back unchanged, so a
    conversation that used thinking is unreplayable if the block is refused — and no template
    in circulation renders a previous turn's reasoning, so it is dropped rather than written
    into the prompt as prose about a question already answered. The echo is what says which
    of the two happened."""
    reply = client.messages.create(
        model=ECHO,
        max_tokens=BUDGET,
        messages=[
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "deliberating", "signature": ""},
                    {"type": "text", "text": "Hi."},
                ],
            },
            {"role": "user", "content": "Again"},
        ],
    )

    assert only_text(reply) == rendered(("user", "Hello"), ("assistant", "Hi."), ("user", "Again"))


def test_thinking_says_what_the_template_is_asked_for() -> None:
    """This dialect names states and not levels, so the three modes reach the template as the
    switch and never as a rung. A request that named no `thinking` leaves it at `auto` rather
    than off: a checkpoint that reasons by design is not asked to stop because a client left
    the field out.

    Off HTTP because the only observable difference is the `Chat` the engine is handed — the
    template on this stand reads no kwarg, and one that did would be testing jinja."""

    def effort(request: dialect.MessagesRequest) -> str:
        return dialect._conversation(request, None).reasoning_effort

    assert effort(body()) == "auto"
    assert effort(body(thinking={"type": "adaptive"})) == "on"
    assert effort(body(thinking={"type": "enabled"})) == "on"
    assert effort(body(thinking={"type": "disabled"})) == "off"


def test_a_profile_names_the_rung_this_dialect_cannot() -> None:
    """The effort lives whole on the profile, and a request that names no `thinking` falls to
    it. One that does throws the switch itself — a state the client named beats a level it
    never asked about."""
    assert dialect._conversation(body(), None, "high").reasoning_effort == "high"
    named = dialect._conversation(body(thinking={"type": "disabled"}), None, "high")
    assert named.reasoning_effort == "off"


def test_the_thinking_budget_is_the_ids_the_block_may_spend() -> None:
    """`budget_tokens` reaches the engine as the reasoning budget, and the profile fills the
    silence. With `disabled` it is refused rather than dropped: there is no block to bound."""
    preset = Sampling(reasoning_budget=64)
    asked = body(thinking={"type": "enabled", "budget_tokens": 512})
    assert dialect._options(asked, preset, None).reasoning_budget == 512
    assert dialect._options(body(), preset, None).reasoning_budget == 64
    assert dialect._options(body(), Sampling(), None).reasoning_budget is None
    with pytest.raises(ValidationError):
        body(thinking={"type": "disabled", "budget_tokens": 512})


def test_context_management_is_honoured_where_it_is_already_true_and_refused_where_it_is_not(
    stand: Stand,
) -> None:
    """`clear_thinking_20251015` asks for something this dialect does by construction — no
    thinking block ever enters a prompt here — so it is accepted and the conversation renders
    the same either way. `clear_tool_uses_20250919` is decided by token thresholds against a
    prompt that only exists inside the engine, so it is refused by name: dropping the edit
    would hand the model the conversation the client asked to have shortened."""
    cleared = post(stand, context_management={"edits": [{"type": "clear_thinking_20251015"}]})
    assert cleared.status_code == 200
    blocks = entry(cleared.json())["content"]
    assert isinstance(blocks, list), blocks
    assert text(entry(blocks[0])["text"]) == rendered(("user", "Hello"))

    refused = post(stand, context_management={"edits": [{"type": "clear_tool_uses_20250919"}]})
    assert refused.status_code == 400
    kind, message = envelope(refused.json())
    assert kind == "invalid_request_error"
    assert "context_management" in message


def test_count_tokens_answers_about_the_rendered_prompt(client: anthropic.Anthropic) -> None:
    """The count is of the prompt the template wrote and not of the turns: the markers, the
    system turn the field became, the generation prompt at the end — all of it is what a
    request pays for, and a sum over the messages would be a number nobody is charged.

    The model has to be resident first, which is what the generation above makes it."""
    ask(client, "Hello", model=COUNTED)

    plain = client.messages.count_tokens(model=COUNTED, messages=turns("Hello"))
    assert plain.input_tokens == len(rendered(("user", "Hello")))

    with_system = client.messages.count_tokens(
        model=COUNTED, messages=turns("Hello"), system="You are terse."
    )
    assert with_system.input_tokens == len(
        rendered(("system", "You are terse."), ("user", "Hello"))
    )


def test_count_tokens_refuses_a_model_that_is_not_resident(client: anthropic.Anthropic) -> None:
    """A load is seconds and tens of gigabytes, asked for on purpose through the residency
    route and never as the side effect of a count — `/admin/models/{id}/tokenize`'s decision,
    and the same one here. The name is in the message so a client knows what to load."""
    with pytest.raises(anthropic.APIStatusError) as raised:
        client.messages.count_tokens(model=CATALOGUED, messages=turns("Hello"))

    assert raised.value.status_code == 409
    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert CATALOGUED in message


def test_the_models_route_answers_for_one_id_and_refuses_a_name_nothing_serves(
    client: anthropic.Anthropic,
) -> None:
    """The listing's single entry, by the name it answers to — slashes and all, which is why
    the path matches them. A name nothing serves comes back `not_found_error` and not an empty
    object: the SDK reads this route to decide whether a model exists."""
    found = client.models.retrieve(CATALOGUED)
    assert found.id == CATALOGUED
    assert found.display_name == CATALOGUED

    with pytest.raises(anthropic.NotFoundError) as raised:
        client.models.retrieve("vendor/nothing")
    kind, message = envelope(raised.value.body)
    assert kind == "not_found_error"
    assert "vendor/nothing" in message
