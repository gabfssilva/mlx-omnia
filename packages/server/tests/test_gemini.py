"""Gate 37.3: the official `google-genai` SDK against a real server process.

The judge is the SDK because this dialect's client does more than read a status: it *parses*
the error body, and it accumulates a stream whose frames it decodes one by one. `TestClient`
and `ASGITransport` run a response to completion before handing it over, so the stream would
be judged after the fact — the server here is a real one on a real port, like the jobs and
metrics suites.

The model under the engine answers with the prompt the checkpoint's own template rendered.
That is the only place the conversion this stage owns is visible: which turn each `content`
became, where a `systemInstruction` went and which name resolved to a checkpoint all exist
inside the render, and the render is on the other side of `stream`.
"""

import json
import socket
import threading
import time
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

import httpx
import mlx.core as mx
import pytest
import uvicorn
from fastapi import FastAPI
from google import genai
from google.genai import errors, types

from mlx_omnia import (
    TEXT,
    Chat,
    ChatCapability,
    ChatMessage,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
    greedy,
)
from mlx_omnia.generate import Constraint
from mlx_omnia.parsers import FALLBACK, Segment, Segmenter
from mlx_omnia.schema import json_instruction
from mlx_omnia_server import catalog, gemini
from mlx_omnia_server.engine import Engine, Job, Loader
from mlx_omnia_server.profiles import Sampling
from mlx_omnia_server.responses import ToolTurn
from mlx_omnia_server.store import Profile, Store

MODEL = "stand/echo"
"""A `/` in the id, like every Hub repository: the name travels in the path, so the route has
to match a tail that carries slashes."""

BASE = "stand/base"
"""No chat template, like a base checkpoint: a conversation has nowhere to go."""

CACHED = "stand/cached"
"""The echo, reporting a prefix reuse. What the dialect writes about a reuse is not what a
trie does to get one, and pinning the number is what keeps the two apart."""

REUSED = 4
"""Rows `CACHED` reports as covered, smaller than the shortest render here."""

WRITER = "stand/json-writer"
PROSE = "stand/prose"
GUIDED = "stand/guided"
"""The three a structured answer is read off. `WRITER` writes what a model asked for JSON
writes anyway — a line of prose, a fence, and the document across two pieces; `PROSE` writes
no JSON at all; and `GUIDED` is the one model here a grammar can be built over, by fiat in
`Recording` below. Whether a checkpoint obeys a schema is the checkpoint's business, and what
these tests are about is what the route does with the answer either way."""

DOCUMENT: dict[str, object] = {"city": "Paris"}

WRITTEN = ("Sure! ", '```json\n{"city": ', '"Paris"}\n```')

CALLER = "stand/caller"
MUTE = "stand/mute"
STRANGER = "stand/stranger"
FLAKY = "stand/flaky"
"""Scripted callers: what a checkpoint writes when it is offered a function is the
checkpoint's own decision, so the generation is pinned and everything around it — the
template, the segmentation, the frames — stays real. `MUTE` writes the call and nothing else,
which is the answer that has to arrive with no text part at all; `STRANGER` writes the same
call behind a template that spells no envelope this server parses."""

PROFILE = "terse"
SYSTEM = "Be terse."

_TEMPLATE = (
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>\n"
    "{% endif %}"
    "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}\n"
    "{% for call in message.tool_calls %}"
    "<tool_call>{{ call.function.name }}{{ call.function.arguments }}\n"
    "{% endfor %}{% endfor %}"
)
"""One line per turn, the role first, and a line per call after the turn that made it. Not a
checkpoint's — what is being read back is the conversation the dialect built, and a real
template would spell it in special tokens. A call is spelled Qwen's way because that spelling
is also what says which family this checkpoint speaks: `parser_of` reads the template's
source, not the generated text, so a stand whose template spells no envelope has no tool
channel at all."""

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

DESCRIPTION = "Current weather in a city"

ENTRIES: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {"name": "get_weather", "description": DESCRIPTION, "parameters": SCHEMA},
    },
)
"""The same function in `chat/completions`'s nested shape, which is what every template reads
and what a `functionDeclaration` has to become. Spelled out rather than converted: a test that
asked the route for the shape would agree with whatever the route did."""

TOOL_BODY: list[dict[str, object]] = [
    {
        "functionDeclarations": [
            {"name": "get_weather", "description": DESCRIPTION, "parametersJsonSchema": SCHEMA}
        ]
    }
]
"""The same declaration as a body, for the tests that post one directly."""

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


@dataclass(frozen=True)
class Echo:
    """Answers with the prompt it was handed, one line per piece.

    The counts are its own, the way `stream_ids` writes them: one id per character of the
    render, so `promptTokenCount` is a fact about the text that reached the model rather than
    about the text the client sent.
    """

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    fails: bool = False
    """Raises after handing out what it had, which is the one way the worker meets an
    exception: an input the model refuses never becomes a job."""

    reused: int = 0
    """Rows a prefix cache would have covered. A number rather than a real trie: what this
    suite is about is what the dialect writes down, and a reuse that depended on two turns
    rendering identically would make the assertion about the template."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value), self.reused)
        for piece in input.value.splitlines(keepends=True)[: options.max_tokens]:
            meter.token()
            yield Segment("content", piece)
        if self.fails:
            raise RuntimeError("the decode thread gave out")


@dataclass(frozen=True)
class Caller:
    """Answers with a call, in the pieces it was given — and with the result of that call once
    the conversation carries one, which is what tells the second turn of a round trip from the
    first: the result only reaches here if the `functionResponse` part became a turn the
    template rendered."""

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
        result = input.value.partition("<tool>")[2].partition("\n")[0]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=input.value
        )
        for piece in (ANSWERED,) if result == RESULT else self.pieces:
            meter.token()
            yield from segmenter.push(piece)

        yield from segmenter.flush()


@dataclass(frozen=True)
class Script:
    """A model whose generation is fixed text, handed out in the pieces it was given. What a
    checkpoint writes when it is asked for a JSON value is the checkpoint's own decision, so
    the text is pinned here and everything around it stays real."""

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
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=input.value
        )
        for piece in self.pieces:
            meter.token()
            yield from segmenter.push(piece)

        yield from segmenter.flush()


TEMPLATE = ChatTemplate.from_source(_TEMPLATE)

FOREIGN = ChatTemplate.from_source(_TEMPLATE.replace("tool_call", "call"))
"""The same template with a call spelled in no family's marker, which is what leaves
`parser_of` with nothing to say and the tool channel shut."""


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id == MODEL:
        return CompositeModel(Echo(), [ChatCapability(TEMPLATE)])
    if model_id == BASE:
        return CompositeModel(Echo(), [])
    if model_id == CACHED:
        return CompositeModel(Echo(reused=REUSED), [ChatCapability(TEMPLATE)])
    if model_id == CALLER:
        return CompositeModel(Caller(CALL_PIECES), [ChatCapability(TEMPLATE)])
    if model_id == MUTE:
        return CompositeModel(Caller(CALL_PIECES[1:]), [ChatCapability(TEMPLATE)])
    if model_id == FLAKY:
        return CompositeModel(Echo(fails=True), [ChatCapability(TEMPLATE)])
    if model_id == STRANGER:
        return CompositeModel(Caller(CALL_PIECES), [ChatCapability(FOREIGN)])
    if model_id == WRITER:
        return CompositeModel(Script(WRITTEN), [ChatCapability(TEMPLATE)])
    if model_id == PROSE:
        return CompositeModel(Script(("I would rather not.",)), [ChatCapability(TEMPLATE)])
    if model_id == GUIDED:
        return CompositeModel(Script((json.dumps(DOCUMENT),)), [ChatCapability(TEMPLATE)])
    raise ValueError(f"no model {model_id!r} in this stand")


class Free:
    """A walk that forbids nothing.

    No model on this stand holds a token table, so a real `Vocabulary` cannot be built over
    one and every schema would end in the same refusal. What a constrained request has to
    prove here is the wiring — that the route compiles the schema and hands the walk to the
    generation — and that is what this stands in for.
    """

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return logits

    def accept(self, token: int) -> bool:
        return True


class Recording(Engine):
    """Keeps the jobs it hands out, and the schemas it was asked to compile. What the dialect
    made of the sampling knobs is the engine's input and reaches no response body.

    Only `GUIDED` gets the double above: every other id falls through to the engine's own
    `constrain`, which is what makes the refusal below a real one — nothing under a scripted
    model has a tokenizer, a head width or a stop id to compile against.
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


@dataclass(frozen=True)
class Stand:
    base_url: str
    engine: Recording
    store: Store


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    engine = Recording(loader)
    store = Store(tmp_path_factory.mktemp("state") / "server.db")
    store.save_profile(
        Profile(
            model=MODEL,
            name=PROFILE,
            sampling=Sampling(temperature=0.0).model_dump_json(exclude_none=True),
            system_prompt=SYSTEM,
        )
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    # The dialect's router alone, on an app of this suite's own: `create_app` is another
    # agent's file this wave, and what is under test is this module's routes.
    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.include_router(gemini.router)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield Stand(base_url=f"http://127.0.0.1:{port}", engine=engine, store=store)
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


def test_the_sdk_round_trips_one_generation(client: genai.Client) -> None:
    """The answer is the render, so what the dialect turned the request into is what comes
    back. The counts are of that render — a `promptTokenCount` taken from the message the
    client sent would be a smaller number, and neither the template nor the count exists
    outside the engine."""
    answer = client.models.generate_content(model=MODEL, contents="Hi")

    rendered = "<user>Hi\n"
    assert answer.text == rendered
    assert candidate(answer).finish_reason == types.FinishReason.STOP
    assert answer.model_version == MODEL
    usage = answer.usage_metadata
    assert usage is not None
    assert usage.prompt_token_count == len(rendered)
    assert usage.candidates_token_count == 1
    assert usage.total_token_count == len(rendered) + 1
    assert usage.cached_content_token_count == 0, "absent would read as a server without it"


def test_a_reused_prefix_is_a_subset_of_the_prompt_count(client: genai.Client) -> None:
    """Here — and in the OpenAI dialect, unlike the Anthropic one — the cached count is part
    of the prompt count and not beside it: `promptTokenCount` stays the whole render, and
    `cachedContentTokenCount` says how much of it the trie handed over."""
    answer = client.models.generate_content(model=CACHED, contents="Hi")

    usage = answer.usage_metadata
    assert usage is not None
    assert usage.cached_content_token_count == REUSED
    assert usage.prompt_token_count == len("<user>Hi\n")
    assert usage.total_token_count == len("<user>Hi\n") + 1


def test_a_model_turn_becomes_an_assistant_turn(client: genai.Client) -> None:
    """`model` is this dialect's spelling and nobody else's: a role passed through untouched
    would render as `<model>`, and a template that has never seen the word renders the turn
    as if it were the user's."""
    answer = client.models.generate_content(
        model=MODEL,
        contents=[
            types.UserContent("one"),
            types.ModelContent("two"),
            types.UserContent("three"),
        ],
    )

    assert answer.text == "<user>one\n<assistant>two\n<user>three\n"


def test_the_system_instruction_is_the_first_turn(client: genai.Client) -> None:
    """The field is the request's own, next to `contents` rather than inside them, so nothing
    puts it in the conversation but this conversion."""
    answer = client.models.generate_content(
        model=MODEL,
        contents="Hi",
        config=types.GenerateContentConfig(system_instruction="Be brief."),
    )

    assert answer.text == "<system>Be brief.\n<user>Hi\n"


def test_the_stream_says_in_pieces_what_the_one_answer_says_whole(client: genai.Client) -> None:
    """Judged by the SDK's own reader: it decodes every frame it is given, so a frame in the
    wrong shape raises there instead of reading as an empty chunk. The finish reason and the
    counts ride the last frame only — a client reading usage off any chunk would be taking a
    partial count for the total."""
    chunks = list(
        client.models.generate_content_stream(
            model=MODEL,
            contents=[types.UserContent("one"), types.ModelContent("two")],
        )
    )

    assert "".join(chunk.text or "" for chunk in chunks) == "<user>one\n<assistant>two\n"
    assert len(chunks) == 3, "one frame per piece, and the frame that closes them"
    last = chunks[-1]
    assert candidate(last).finish_reason == types.FinishReason.STOP
    assert last.usage_metadata is not None
    assert last.usage_metadata.candidates_token_count == 2
    assert all(candidate(chunk).finish_reason is None for chunk in chunks[:-1])
    assert all(chunk.usage_metadata is None for chunk in chunks[:-1])


def test_max_output_tokens_reaches_the_engine(client: genai.Client) -> None:
    """The one sampling knob a scripted model can answer for: cut the budget and the answer is
    shorter. A `generationConfig` that never left the request model would give three lines."""
    answer = client.models.generate_content(
        model=MODEL,
        contents=[
            types.UserContent("one"),
            types.ModelContent("two"),
            types.UserContent("three"),
        ],
        config=types.GenerateContentConfig(max_output_tokens=2),
    )

    assert answer.text == "<user>one\n<assistant>two\n"


def test_a_name_with_a_slash_and_a_colon_routes_to_the_checkpoint_and_the_profile(
    client: genai.Client,
) -> None:
    """Both separators in one URL: `models/stand/echo:terse:generateContent`. The method is the
    last colon's, the profile the one before it, and the slash is part of the name — the stand's
    loader answers to `stand/echo` and to nothing else, so a split made anywhere else is a
    request that never reaches a model."""
    answer = client.models.generate_content(model=f"{MODEL}:{PROFILE}", contents="Hi")

    assert answer.text == f"<system>{SYSTEM}\n<user>Hi\n"
    assert answer.model_version == f"{MODEL}:{PROFILE}"


def test_the_profile_fills_the_knob_the_request_left_out(
    stand: Stand, client: genai.Client
) -> None:
    """No dialect has a field for a profile, so the name selects it and the request overrides
    it. `temperature: 0` is the profile's, and the deterministic end of the dial is the one
    sampler that can be named from out here."""
    client.models.generate_content(model=f"{MODEL}:{PROFILE}", contents="Hi")
    assert stand.engine.jobs[-1].options.sampler is greedy

    client.models.generate_content(
        model=f"{MODEL}:{PROFILE}",
        contents="Hi",
        config=types.GenerateContentConfig(temperature=1.0),
    )
    assert stand.engine.jobs[-1].options.sampler is not greedy


def test_an_unknown_method_after_the_colon_is_this_dialect_s_error(client: genai.Client) -> None:
    """`count_tokens` is a real method of the SDK and not one this server serves, so it walks
    the same path with a tail the router would otherwise answer 404 to with a body of its own
    shape. What the assertion is about is the message: `APIError` parses `status`, `message`
    and `code` out of the envelope, and prints `None None.` when they are not there."""
    with pytest.raises(errors.ClientError) as raised:
        client.models.count_tokens(model=MODEL, contents="Hi")

    failure = raised.value
    assert failure.code == 404
    assert failure.status == "NOT_FOUND"
    assert failure.message is not None and "countTokens" in failure.message
    assert "None None" not in str(failure)


def test_an_unknown_model_is_a_client_error_the_sdk_can_read(client: genai.Client) -> None:
    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(model="nope", contents="Hi")

    failure = raised.value
    assert failure.code == 404
    assert failure.status == "NOT_FOUND"
    assert failure.message is not None and "nope" in failure.message
    assert "None None" not in str(failure)


def test_a_model_without_a_chat_template_is_refused(client: genai.Client) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go: the
    client's error, named, and not a 500 out of the worker."""
    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(model=BASE, contents="Hi")

    failure = raised.value
    assert failure.code == 400
    assert failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "chat template" in failure.message


def test_a_knob_this_dialect_does_not_honour_is_refused_by_name(stand: Stand) -> None:
    """Accepting `stopSequences` and never cutting on one tells the client it was applied. The
    envelope a refused body comes back in is the app's handler, and `test_dialect_errors.py`
    is where that is read — what this asserts is that it is refused at all, and that the field
    is named where the client can read it."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": "Hi"}]}],
            "generationConfig": {"stopSequences": ["x"]},
        },
        timeout=30,
    )

    assert 400 <= response.status_code < 500, response.text
    assert "stopSequences" in response.text


def test_a_content_with_no_role_is_a_user_turn(stand: Stand) -> None:
    """What the API documents for an absent role. The SDK always writes one, so nothing above
    reaches the default — and a turn that quietly became the assistant's would put words in the
    model's mouth."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={"contents": [{"parts": [{"text": "Hi"}]}]},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "<user>Hi\n"


ASKED = "Weather in Paris?"


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


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(
    stand: Stand, client: genai.Client
) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function and
    answers with a `functionCall` part beside its text, the result goes back as the
    `functionResponse` part of the next content, and the second answer is one only a model that
    was handed the result can give.

    The conversation is compared against the one `chat/completions` builds out of the same
    round. Same characters into the model through two dialects — which is what makes one
    checkpoint answer both the same way, and what a `functionResponse` keyed by name instead of
    by id has to survive.
    """
    config = offered()
    first = client.models.generate_content(model=CALLER, contents=ASKED, config=config)

    assert first.text == PREAMBLE
    calls = first.function_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].args == {"city": "Paris"}
    assert candidate(first).finish_reason == types.FinishReason.STOP

    second = client.models.generate_content(
        model=CALLER,
        contents=[
            types.UserContent(ASKED),
            types.Content(
                role="model",
                parts=[
                    types.Part(text=PREAMBLE),
                    types.Part(
                        function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"})
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="get_weather", response={"output": RESULT}
                    )
                ],
            ),
        ],
        config=config,
    )

    assert second.text == ANSWERED
    expected: tuple[ToolTurn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": PREAMBLE,
            "tool_calls": [
                {
                    "id": "get_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ARGUMENTS},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "get_weather"},
    )
    built = Chat(expected, tools=ENTRIES)
    assert submitted(stand) == built
    # And the same characters, which the comparison above cannot see: a template renders a
    # tool entry with `tojson`, and two dicts that differ only in the order their keys went
    # in are equal and do not write the same prompt.
    assert TEMPLATE.render(submitted(stand)) == TEMPLATE.render(built)


def test_the_call_rides_the_frame_that_closes_the_stream(client: genai.Client) -> None:
    """The call whole in one terminal frame — A11's decision — and that frame is the one the
    finish reason and the counts already ride: a client reading a completed turn reads it
    there. The text before it arrives piece by piece, envelope suppressed, which is what says
    the reader ran over the stream and not over the finished answer."""
    chunks = list(
        client.models.generate_content_stream(model=CALLER, contents=ASKED, config=offered())
    )

    assert "".join(chunk.text or "" for chunk in chunks) == PREAMBLE
    assert all(chunk.function_calls is None for chunk in chunks[:-1])
    last = chunks[-1]
    calls = last.function_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].args == {"city": "Paris"}
    assert candidate(last).finish_reason == types.FinishReason.STOP
    assert last.usage_metadata is not None


def test_a_content_that_only_called_something_carries_no_text_part(
    stand: Stand, client: genai.Client
) -> None:
    """The empty text part this dialect does not write: a part carrying `""` beside a call is a
    model that answered with nothing before it called, and the SDK's `.text` reads it as an
    answer. The stream says the same thing — one frame, the call's."""
    answer = client.models.generate_content(model=MUTE, contents=ASKED, config=offered())

    content = candidate(answer).content
    assert content is not None and content.parts is not None
    assert [part.text for part in content.parts] == [None]
    assert answer.function_calls is not None and len(answer.function_calls) == 1

    streamed = httpx.post(
        url(stand, f"{MUTE}:streamGenerateContent"),
        json={"contents": [{"parts": [{"text": ASKED}]}], "tools": TOOL_BODY},
        timeout=30,
    )
    frames = [line for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert len(frames) == 1, frames
    assert "functionCall" in frames[0]
    assert '"text"' not in frames[0]


def test_mode_none_neither_offers_the_declarations_nor_reads_a_call_back(
    stand: Stand, client: genai.Client
) -> None:
    """`NONE` is honoured where it can be honoured: the declarations never reach the prompt, so
    the model has nothing to call rather than an instruction not to. And a turn nobody was
    offered a function for is text — answering with a call would be answering with the one
    thing the client asked us not to do."""
    config = offered(types.FunctionCallingConfigMode.NONE)
    answer = client.models.generate_content(model=MUTE, contents=ASKED, config=config)

    assert answer.text == ENVELOPE
    assert answer.function_calls is None
    assert submitted(stand).tools == ()


def test_a_checkpoint_whose_envelope_nothing_here_parses_answers_with_the_text(
    client: genai.Client,
) -> None:
    """Which family a checkpoint speaks is a fact of its chat template and not of the text it
    writes: read off the output instead, Qwen3.6's `<tool_call><function=…>` is taken for Qwen's
    JSON and the envelope is held for a parser that cannot read it. A template that spells none
    leaves the channel shut, and what the model wrote reaches the client whole."""
    answer = client.models.generate_content(model=STRANGER, contents=ASKED, config=offered())

    assert answer.text == PREAMBLE + ENVELOPE
    assert answer.function_calls is None


def test_forcing_a_call_is_refused_by_name(stand: Stand) -> None:
    """`ANY` and `VALIDATED` constrain decoding to a call and there is no such constraint here:
    answering `AUTO` to a client that asked for one is a call the model may never have made.
    The envelope a refused body comes back in is 37.1's — what this asserts is that it is
    refused at all, and that the field is named where the client can read it."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "tools": TOOL_BODY,
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        },
        timeout=30,
    )

    assert 400 <= response.status_code < 500, response.text
    assert "mode" in response.text


@pytest.mark.parametrize("spelling", ["parametersJsonSchema", "parameters_json_schema"])
def test_a_json_schema_reaches_the_prompt_under_either_of_its_two_names(
    stand: Stand, spelling: str
) -> None:
    """Proto's JSON mapping accepts a field's own name as well as its camelCase form: the REST
    reference documents `parametersJsonSchema` and the SDK sends `parameters_json_schema`, so a
    dialect that took one of them refuses half the clients that speak it. Both arrive at the
    template as `parameters`, which is the only field a template has."""
    declaration: dict[str, object] = {
        "name": "get_weather",
        "description": DESCRIPTION,
        spelling: SCHEMA,
    }
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "tools": [{"functionDeclarations": [declaration]}],
        },
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert submitted(stand).tools == ENTRIES


def test_the_sdks_own_schema_travels_untouched(stand: Stand) -> None:
    """`parameters` is the other spelling, and what rides it is the OpenAPI subset the SDK
    builds out of its `Schema`: types spelled `OBJECT` and `STRING`. It reaches the model as
    written — normalizing a client's schema is rewriting what it declared — and a declaration
    with no description carries none rather than a null."""
    schema: dict[str, object] = {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}}
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={
            "contents": [{"parts": [{"text": ASKED}]}],
            "tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": schema}]}],
        },
        timeout=30,
    )

    assert response.status_code == 200, response.text
    assert submitted(stand).tools == (
        {"type": "function", "function": {"name": "get_weather", "parameters": schema}},
    )


def test_the_models_list_carries_the_catalog_and_every_name_it_answers_to(
    stand: Stand, client: genai.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog, not the residents, under this dialect's name for a model — `models/{id}`
    is what the SDK puts back into the path. A profile is listed as a model of its own because
    the name is the only field a client has to select one with.

    The scan is replaced rather than read: it walks the machine's own Hub cache, and what this
    asserts is the shape and the profile, not what someone happened to have downloaded.
    """
    entry = catalog.CatalogEntry(
        id=MODEL,
        directory=Path("/nowhere"),
        store=Path("/nowhere"),
        architecture="echo",
        quantization=None,
        dtype=None,
        context=None,
        bytes_on_disk=0,
    )
    monkeypatch.setattr(catalog, "scan", lambda: [entry])

    listed = list(client.models.list())

    assert [model.name for model in listed] == [f"models/{MODEL}", f"models/{MODEL}:{PROFILE}"]
    assert listed[0].supported_actions == ["generateContent", "streamGenerateContent"]


def test_a_generation_that_dies_mid_stream_raises_instead_of_closing_with_stop(
    client: genai.Client,
) -> None:
    """The failure a stream can hide. The response is already 200 and the frames already
    going out when the decode thread gives out; the closing frame this dialect writes carries
    `finishReason: STOP` and a `usageMetadata` — a failure wearing the shape of an answer,
    which the SDK would decode without a word. It reads a chunk opening with `error` and
    raises on that, so the failure travels as one."""
    with pytest.raises(errors.APIError) as failure:
        for _ in client.models.generate_content_stream(model=FLAKY, contents="one"):
            pass

    assert "RuntimeError" in str(failure.value), "the reason is the daemon's own words"


def test_a_generation_the_budget_cut_says_max_tokens_and_not_stop(client: genai.Client) -> None:
    """`MAX_TOKENS` is the branch an agent loop takes to continue. `STOP` over a sentence
    `maxOutputTokens` cut is a truncation reported as a finished answer, and nothing else in
    the body says otherwise. Both shapes, because the reason rides the closing frame here."""
    whole = client.models.generate_content(
        model=MODEL,
        contents=[types.UserContent("one"), types.ModelContent("two")],
        config=types.GenerateContentConfig(max_output_tokens=1),
    )
    assert candidate(whole).finish_reason == types.FinishReason.MAX_TOKENS

    chunks = list(
        client.models.generate_content_stream(
            model=MODEL,
            contents=[types.UserContent("one"), types.ModelContent("two")],
            config=types.GenerateContentConfig(max_output_tokens=1),
        )
    )
    assert candidate(chunks[-1]).finish_reason == types.FinishReason.MAX_TOKENS


def test_a_generation_that_ended_on_its_own_still_says_stop(client: genai.Client) -> None:
    """The other half, and what keeps the one above from asserting that every answer is
    truncated: the same model under a budget it does not reach."""
    answer = client.models.generate_content(
        model=MODEL,
        contents=[types.UserContent("one")],
        config=types.GenerateContentConfig(max_output_tokens=64),
    )
    assert candidate(answer).finish_reason == types.FinishReason.STOP


def test_a_json_mime_type_is_checked_and_comes_back_as_the_document(
    stand: Stand, client: genai.Client
) -> None:
    """Level 1 in the half of it this dialect can ask for: `application/json` with no schema
    beside it is the demand that the answer be JSON, and it is checked once the generation is
    spent — upstream the same field says the model "needs to be prompted", which is exactly
    what the instruction turn is.

    The prose and the fence stay out of the answer: a client that asked for JSON and got
    `Sure! ```json…``` ` back was handed something `json.loads` refuses, and the document is
    the one thing this route can promise — it is what it validated."""
    answer = client.models.generate_content(
        model=WRITER,
        contents=ASKED,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    assert answer.text is not None
    assert json.loads(answer.text) == DOCUMENT
    assert "Sure!" not in answer.text
    turns: tuple[ChatMessage, ...] = (
        {"role": "user", "content": ASKED},
        {"role": "system", "content": json_instruction(None)},
    )
    assert submitted(stand).messages == turns


def test_an_answer_with_no_json_in_it_is_the_daemon_s_failure_and_says_so(
    client: genai.Client,
) -> None:
    """The one way level 1 fails here. INTERNAL and not INVALID_ARGUMENT: what did not
    validate is what this server generated, and of the four statuses this dialect has, that is
    the one that does not blame the client for it. The SDK retries nothing by default, so the
    generation is not bought again behind the client's back."""
    with pytest.raises(errors.ServerError) as raised:
        client.models.generate_content(
            model=PROSE,
            contents=ASKED,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

    failure = raised.value
    assert failure.code == 500
    assert failure.message is not None and "no JSON value" in failure.message
    assert "None None" not in str(failure)


def test_a_stream_checks_what_it_already_sent_and_raises_at_the_client(
    client: genai.Client,
) -> None:
    """A stream has one pass: the frames are gone by the time the document can be checked, so
    the violation travels the way a generation that died travels — a chunk opening with
    `error`, which the SDK reads and raises on. The closing frame this dialect writes carries
    `finishReason: STOP` and a `usageMetadata`, and sending it here would be a failure wearing
    the shape of an answer."""
    with pytest.raises(errors.APIError) as raised:
        for _ in client.models.generate_content_stream(
            model=PROSE,
            contents=ASKED,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        ):
            pass

    assert "no JSON value" in str(raised.value)


def test_a_schema_is_a_guarantee_here_and_reaches_the_generation_as_a_walk(
    stand: Stand, client: genai.Client
) -> None:
    """This dialect has no `strict` to turn a schema into a check, because upstream a schema is
    constrained decoding: what a client sends `responseJsonSchema` for is an answer that cannot
    violate it. So the schema is compiled against the model the request named and the walk that
    comes back is what the generation runs under — a route that compiled it and dropped the
    walk would answer 200 with a free decode, which is the guarantee broken silently.

    Nothing goes into the prompt: the mask is the whole of it, and a schema in the prompt as
    well would be paying for the answer twice."""
    answer = client.models.generate_content(
        model=GUIDED,
        contents=ASKED,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=SCHEMA
        ),
    )

    assert answer.parsed == DOCUMENT, "the SDK reads its own field back off the answer"
    assert stand.engine.compiled[-1] == SCHEMA
    assert stand.engine.jobs[-1].options.constraint is not None
    alone: tuple[ChatMessage, ...] = ({"role": "user", "content": ASKED},)
    assert submitted(stand).messages == alone


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    client: genai.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model of this stand is in no catalog and holds no tokenizer, so there is no token
    table to compile against. What the client must not get is the answer anyway: a schema here
    asks for a guarantee, and a 200 carrying a free decode is that guarantee broken silently.

    The catalog is emptied for the read the engine makes on the way to that refusal: it walks
    the machine's own Hub cache, and what this asserts has nothing to do with what someone
    happened to have downloaded."""
    monkeypatch.setattr(catalog, "scan", list)

    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(
            model=MODEL,
            contents=ASKED,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=SCHEMA
            ),
        )

    failure = raised.value
    assert failure.code == 400
    assert failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "token table" in failure.message


def test_the_sdks_own_response_schema_is_refused_by_name(client: genai.Client) -> None:
    """`response_schema` is the other spelling, and what rides it is the OpenAPI subset the SDK
    builds out of its `Schema`: types spelled `OBJECT` and `STRING`, keys `property_ordering`.
    What is compiled into a grammar here is a JSON Schema, and translating one spelling into
    the other would be a second reading of a vocabulary this server does not own — a grammar
    subtly unlike the one the client asked for is worse than a refusal that names the field
    that carries it.

    A fresh dict rather than `SCHEMA`: `t_schema` processes what it is handed in place."""
    with pytest.raises(errors.ClientError) as raised:
        client.models.generate_content(
            model=MODEL,
            contents=ASKED,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        )

    failure = raised.value
    assert failure.code == 400
    assert failure.message is not None and "responseJsonSchema" in failure.message


@pytest.mark.parametrize(
    ("body", "named"),
    [
        ({"generationConfig": {"responseJsonSchema": SCHEMA}}, "responseMimeType"),
        (
            {
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": SCHEMA,
                },
                "tools": TOOL_BODY,
            },
            "tools",
        ),
    ],
    ids=["without the mime type", "with tools"],
)
def test_the_two_shapes_a_schema_cannot_be_honoured_in(
    stand: Stand, body: dict[str, object], named: str
) -> None:
    """A schema with no mime type beside it is the API's own rule — it asks for a document and
    for prose around it at the same time. A schema with tools is the combination the mask makes
    impossible: it allows the schema's ids from the first token, so an offered function can
    never be called, and a 200 with the tools silently uncallable is the answer this refusal
    exists instead of.

    Posted raw, because the SDK is not what is under test here: both bodies are ones a client
    can write, and what is asserted is that each is refused with the field that was wrong in
    the message."""
    response = httpx.post(
        url(stand, f"{MODEL}:generateContent"),
        json={"contents": [{"parts": [{"text": ASKED}]}], **body},
        timeout=30,
    )

    assert response.status_code == 400, response.text
    assert named in response.json()["error"]["message"]
