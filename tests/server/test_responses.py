"""`/api/openai/v1/responses` judged by the official SDK against a real server process.

The judge matters more here than in the chat dialect. A Responses stream is a sequence of
*named* frames whose only consumer is an accumulator: it refuses anything before
`response.created`, indexes each delta into an item and a content part it was told about
first, and rebuilds the whole `Response` at the end. Asserting on the JSON this route writes
would be asserting that it writes what this file thinks it should; `client.responses.stream`
is what says whether the frames add up to an answer.

The model under the engine is scripted and the chat template is written out below, so the
prompt is a string this file can predict. What is under test is the mapping from the
dialect's shapes — a bare string, a list of typed items, `instructions` as a field of its
own — to the conversation the template renders, and none of it depends on which checkpoint
happens to be in the cache.
"""

import json
import socket
import threading
import time
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeIs

import httpx
import mlx.core as mx
import pytest
import uvicorn
from fastapi import FastAPI
from openai import BadRequestError, NotFoundError, OpenAI, UnprocessableEntityError
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseTextConfigParam,
)

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
)
from mlx_omnia import ChatMessage as Turn
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.engine.parsers import FALLBACK, Segment, Segmenter
from mlx_omnia.engine.schema import json_instruction
from mlx_omnia.server import catalog
from mlx_omnia.server import responses as responses_module
from mlx_omnia.server.engine import Engine, Loader
from mlx_omnia.server.profiles import Sampling
from mlx_omnia.server.responses import router
from mlx_omnia.server.store import Profile, Store

CACHED = "cached"
"""The scripted model, reporting a prefix reuse. What the dialect writes about a reuse is not
what a trie does to get one, and pinning the number is what keeps the two apart."""

REUSED = 5
"""Rows `CACHED` reports as covered, smaller than the shortest render here."""

SCRIPTED = "scripted"
BROKEN = "broken"
SLOW = "slow"
BARE = "bare"
"""A model with no chat template: a conversation is not an input it takes."""

CALLER = "caller"
MUTE = "mute"
CUT = "cut"
STRANGER = "stranger"
"""Scripted callers. What a checkpoint writes when it is offered a function is the
checkpoint's own decision, so the generated text is pinned here and everything around it —
the template, the segmentation, the frames — stays real. `MUTE` writes the call and nothing
else, `CUT` half an envelope, and `STRANGER` the whole one behind a template that spells no
envelope this server parses."""

WRITER = "writer"
BREAKER = "breaker"
GUIDED = "guided"
"""The three a structured answer is read off. `WRITER` writes what a model asked for JSON
writes anyway — a line of prose, a fence, and the document across two pieces; `BREAKER` a
document that parses and breaks the schema; and `GUIDED` is the one model here a grammar can
be built over, by fiat in `Constrained` below. Whether a checkpoint obeys a schema is the
checkpoint's business, and what these tests are about is what the route does with the answer
either way."""

DOCUMENT: dict[str, object] = {"city": "Paris"}

WRITTEN = ("Sure! ", '```json\n{"city": ', '"Paris"}\n```')
"""Nothing here is malformed — what it is, is not `json.loads`-able as it stands, which is
what the client asked for."""

FAULTY = '{"town": "Paris"}'
"""JSON that parses and does not conform: `$.city is required and missing`."""

PIECES = ("Paris", " is", " the", " capital.")
"""One generation, in the pieces the route has to hand out one delta each: a route that
buffered would produce the same answer and a different stream."""

ANSWER = "".join(PIECES)

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

SOURCE = (
    "{% if tools %}<|tools|>{% for tool in tools %}{{ tool | tojson }}{% endfor %}<|end|>"
    "{% endif %}"
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% for call in message.tool_calls %}"
    "<tool_call>{{ call.function.name }}{{ call.function.arguments }}</tool_call>"
    "{% endfor %}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

TEMPLATE = ChatTemplate.from_source(SOURCE)

FOREIGN = ChatTemplate.from_source(SOURCE.replace("tool_call", "call"))
"""The same template with a call spelled in no family's marker, which is what leaves
`parser_of` with nothing to say and the tool channel shut."""
"""Written out rather than downloaded: what is under test is which turns and which tools the
dialect builds, and a template that spells them back is what makes the prompt readable. A call
is spelled Qwen's way because that spelling is also what says which family this checkpoint
speaks — `parser_of` reads the source, not the generated text — so a stand whose template
spells no envelope has no tool channel at all. The checkpoint's own template is what
`mlx_omnia.load` brings, and that path is `test_api.py`'s."""

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

DESCRIPTION = "Current weather in a city"

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": DESCRIPTION,
        "parameters": SCHEMA,
        "strict": False,
    }
]

CHECKED: ResponseTextConfigParam = {
    "format": {"type": "json_schema", "name": "weather", "schema": SCHEMA}
}
"""Level 1 in this dialect's shape: the name, the schema and `strict` sit *on* the format,
where `chat/completions` nests them under a `json_schema` object. Typed as the SDK types it,
so the same value serves `create(text=...)` and the httpx body."""

GUARANTEED: ResponseTextConfigParam = {
    "format": {"type": "json_schema", "name": "weather", "schema": SCHEMA, "strict": True}
}

ONLY_JSON: ResponseTextConfigParam = {"format": {"type": "json_object"}}

ENTRIES: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {"name": "get_weather", "description": DESCRIPTION, "parameters": SCHEMA},
    },
)
"""The same function in `chat/completions`'s nested shape, which is what every template reads
and what this dialect's flat one has to become. Spelled out rather than converted here: a test
that asked the route for the shape would agree with whatever the route did."""


@dataclass(frozen=True)
class Call:
    prompt: str
    options: GenerationOptions


CALLS: list[Call] = []
"""What reached the model, in order. A test reads the last entry after its own request,
which has finished by the time the answer is back."""


def last() -> Call:
    assert CALLS, "no request ever reached the model"
    return CALLS[-1]


@dataclass(frozen=True)
class Script:
    """A model whose generation is fixed text. It counts through the meter — one mark per
    piece, one prompt token per character — so the usage the dialect reports is the model's
    own numbers rather than a constant that happens to look right."""

    pieces: tuple[str, ...]
    delay: float = 0.0
    fails: bool = False
    reused: int = 0
    """Rows a prefix cache would have covered. A number rather than a real trie: what this
    suite is about is what the dialect writes down, and a reuse that depended on two turns
    rendering identically would make the assertion about the template."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        CALLS.append(Call(input.value, options))
        time.sleep(self.delay)
        meter.prefill(len(input.value), self.reused)
        # A conversation that already carries the result of a call is answered with it, which
        # is what tells the second turn of a round trip from the first: the result reaches
        # here only if the `function_call_output` item became a turn the template rendered.
        result = input.value.partition("<|tool|>")[2].partition("<|end|>")[0]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=input.value
        )
        for piece in (f"It is {result}.",) if result else self.pieces:
            meter.token()
            yield from segmenter.push(piece)
        yield from segmenter.flush()
        if self.fails:
            raise RuntimeError("the model fell over")


def loader(model_id: str) -> LanguageModel[ModelInput]:
    match model_id:
        case "scripted":
            return CompositeModel(Script(PIECES), [ChatCapability(TEMPLATE)])
        case "broken":
            return CompositeModel(Script(("Par",), fails=True), [ChatCapability(TEMPLATE)])
        case "slow":
            delay = responses_module._KEEP_ALIVE_SECONDS * 2
            return CompositeModel(Script(PIECES, delay=delay), [ChatCapability(TEMPLATE)])
        case "bare":
            return CompositeModel(Script(PIECES), [])
        case "cached":
            return CompositeModel(Script(PIECES, reused=REUSED), [ChatCapability(TEMPLATE)])
        case "caller":
            return CompositeModel(Script(CALL_PIECES), [ChatCapability(TEMPLATE)])
        case "mute":
            return CompositeModel(Script(CALL_PIECES[1:]), [ChatCapability(TEMPLATE)])
        case "cut":
            envelope = ('<tool_call>\n{"name": "get_weat',)
            return CompositeModel(Script(envelope), [ChatCapability(TEMPLATE)])
        case "stranger":
            return CompositeModel(Script(CALL_PIECES), [ChatCapability(FOREIGN)])
        case "writer":
            return CompositeModel(Script(WRITTEN), [ChatCapability(TEMPLATE)])
        case "breaker":
            return CompositeModel(Script((FAULTY,)), [ChatCapability(TEMPLATE)])
        case "guided":
            return CompositeModel(Script((json.dumps(DOCUMENT),)), [ChatCapability(TEMPLATE)])
        case other:
            raise ValueError(f"no model {other!r} in this stand")


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

    def __init__(self, loader: Loader) -> None:
        super().__init__(loader)
        self.compiled: list[Mapping[str, object]] = []

    async def constrain(self, model_id: str, schema: Mapping[str, object]) -> Constraint:
        if model_id != GUIDED:
            return await super().constrain(model_id, schema)
        self.compiled.append(schema)
        return Free()


@dataclass
class Stand:
    base_url: str
    store: Store
    engine: Constrained


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    """The route's own router on a throwaway app: `create_app` is being rewritten this wave,
    and what this suite is about is the dialect and not the wiring. A real process, because
    `TestClient` and `ASGITransport` run the whole response before handing it over and half
    of what is asserted below is about when a frame arrives."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine = Constrained(loader)
    store = Store(tmp_path_factory.mktemp("state") / "server.db")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.include_router(router)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield Stand(base_url=f"http://127.0.0.1:{port}", store=store, engine=engine)
    server.should_exit = True
    thread.join(timeout=5)
    assert not thread.is_alive(), "the stand's server did not shut down"


@pytest.fixture(scope="module")
def client(stand: Stand) -> OpenAI:
    return OpenAI(base_url=f"{stand.base_url}/api/openai/v1", api_key="unused")


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


def test_the_sdk_reads_the_answer_and_the_model_s_own_numbers(client: OpenAI) -> None:
    """The whole non-streaming shape at once, because the SDK parses it as one: an `output`
    that is not a list of typed items has no `output_text` at all.

    The usage is asserted against the prompt that actually reached the model, and the budget
    against the options the engine was handed — the two fields the dialect renames on the way
    down, and the two a route can get wrong without the answer looking any different."""
    answer = client.responses.create(
        model=SCRIPTED, input="Where is Paris?", max_output_tokens=7, temperature=0
    )

    assert answer.output_text == ANSWER
    assert answer.status == "completed"
    assert answer.id.startswith("resp_")
    assert answer.model == SCRIPTED
    item = answer.output[0]
    assert item.type == "message"
    assert item.role == "assistant"
    assert item.status == "completed"
    call = last()
    assert call.options.max_tokens == 7
    usage = answer.usage
    assert usage is not None
    assert usage.output_tokens == len(PIECES)
    assert usage.input_tokens == len(call.prompt)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.input_tokens_details.cached_tokens == 0, (
        "absent would read as a server without the field, and zero is a miss"
    )


def test_a_reused_prefix_is_a_subset_of_the_input_count(client: OpenAI) -> None:
    """This dialect counts the way the chat one does and unlike the Anthropic one:
    `input_tokens` stays the whole prompt and `cached_tokens` says how much of it the trie
    handed over, so `total_tokens` is unmoved by a hit. `cache_write_tokens` is zero because
    it is — the trie fills from the forward the turn was running anyway."""
    answer = client.responses.create(model=CACHED, input="Where is Paris?", temperature=0)

    usage = answer.usage
    assert usage is not None
    assert usage.input_tokens_details.cached_tokens == REUSED
    assert usage.input_tokens == len(last().prompt)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


def test_the_named_frames_accumulate_into_the_same_answer(client: OpenAI) -> None:
    """`client.responses.stream` is the judge: its accumulator raises if `response.created`
    is not the first frame, if a delta names an item or a content part it was never told
    about, or if `response.completed` never arrives. Asserted on top of that is the part the
    accumulator tolerates — one delta per piece, in order, and sequence numbers that are a
    counter rather than a constant."""
    with client.responses.stream(model=SCRIPTED, input="Where is Paris?", temperature=0) as stream:
        seen = list(stream)
        final = stream.get_final_response()

    kinds = [event.type for event in seen]
    assert kinds[0] == "response.created"
    assert kinds[1:3] == ["response.output_item.added", "response.content_part.added"]
    assert "response.output_text.done" in kinds
    assert kinds[-1] == "response.completed"
    assert [event.sequence_number for event in seen] == list(range(len(seen)))
    deltas = [event.delta for event in seen if event.type == "response.output_text.delta"]
    assert tuple(deltas) == PIECES
    assert final.output_text == ANSWER
    assert final.status == "completed"
    usage = final.usage
    assert usage is not None and usage.output_tokens == len(PIECES)


def test_instructions_are_the_system_turn_the_chat_dialect_would_have_sent(
    client: OpenAI,
) -> None:
    """`instructions` is a field and not a message, and the checkpoint's template has one
    place to put it. The prompt is compared against the render of the conversation the chat
    dialect builds out of `[system, user]` — the same ids, which is what makes one model
    answer two dialects the same way.

    Written out rather than fetched through `chat/completions`: that handler is another
    agent's file this wave, and what the two dialects have to agree on is the conversation,
    which is this side of it."""
    client.responses.create(
        model=SCRIPTED, input="Where is Paris?", instructions="Answer in one word."
    )

    turns: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Where is Paris?"},
    )
    assert last().prompt == rendered(turns)


def test_a_list_of_items_is_the_conversation_it_spells(client: OpenAI) -> None:
    """The other half of `input`: typed items, including the ones this route wrote itself. An
    assistant turn comes back as `output_text` parts, and a client replaying it carries the
    `id`, `status` and `annotations` the dialect gave it — dropped here, since none of them
    is a turn.

    `developer` is the dialect's newer name for a system turn and the template knows only the
    older one, so a route that passed it through would render a role no checkpoint has."""
    items: ResponseInputParam = [
        {"role": "developer", "content": "Answer in one word."},
        {"role": "user", "content": [{"type": "input_text", "text": "Where is Paris?"}]},
        {
            "id": "msg_replayed",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
        },
        {"role": "user", "content": "And Lyon?"},
    ]
    answer = client.responses.create(model=SCRIPTED, input=items)

    turns: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Where is Paris?"},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": "And Lyon?"},
    )
    assert last().prompt == rendered(turns)
    assert answer.output_text == ANSWER


def test_a_profile_fills_what_the_request_left_out_and_nothing_else(
    stand: Stand, client: OpenAI
) -> None:
    """A profile is selected by the one field every dialect has, the model name. Its system
    prompt gives way to `instructions` for the same reason the chat dialect's gives way to a
    system message: two system turns is a conversation the template has to pick between.

    Its knobs give way the same way, and the request's own default is not what decides it —
    `repetition_penalty` is 1.0 unset, which is exactly the value that turns the penalty off,
    so a route reading the value instead of `model_fields_set` would drop the profile's."""
    stand.store.save_profile(
        Profile(
            model=SCRIPTED,
            name="brief",
            sampling=Sampling(repetition_penalty=1.2).model_dump_json(exclude_none=True),
            system_prompt="You are terse.",
        )
    )

    client.responses.create(model=f"{SCRIPTED}:brief", input="Where is Paris?")
    from_profile: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Where is Paris?"},
    )
    assert last().prompt == rendered(from_profile)
    assert last().options.penalty is not None

    client.responses.create(
        model=f"{SCRIPTED}:brief",
        input="Where is Paris?",
        instructions="Answer in one word.",
        extra_body={"repetition_penalty": 1.0},
    )
    from_request: tuple[ChatMessage, ...] = (
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Where is Paris?"},
    )
    assert last().prompt == rendered(from_request)
    assert last().options.penalty is None


def test_store_is_refused_by_name(client: OpenAI) -> None:
    """Persistent responses are not this server's: an id it handed back would name nothing.
    Refusing by name is the difference between a client that knows and one that keeps a
    conversation id and finds out later."""
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=SCRIPTED, input="Where is Paris?", store=True)

    assert code(raised.value.body) == "store_unsupported"
    # The field is declared so the refusal can be named, which only holds if `false` answers.
    assert client.responses.create(model=SCRIPTED, input="Hi", store=False).output_text == ANSWER


def test_an_empty_input_is_refused_and_the_next_request_is_answered(client: OpenAI) -> None:
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=SCRIPTED, input="")

    assert code(raised.value.body) == "empty_input"
    assert client.responses.create(model=SCRIPTED, input="Hi").output_text == ANSWER


def test_a_model_that_takes_no_conversation_is_refused_by_name(client: OpenAI) -> None:
    """A base model ships no chat template, so nothing turns a conversation into a prompt. It
    is a client error and not a missing model: the checkpoint is there."""
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=BARE, input="Where is Paris?")

    assert code(raised.value.body) == "unsupported_input"


def test_an_unknown_model_is_the_sdk_s_not_found(client: OpenAI) -> None:
    with pytest.raises(NotFoundError) as raised:
        client.responses.create(model="nope", input="Where is Paris?")

    assert code(raised.value.body) == "model_not_found"


def test_a_generation_that_dies_mid_stream_ends_in_response_failed(client: OpenAI) -> None:
    """`response.completed` is the frame the accumulator turns into a final response, so a
    generation that failed must not wear it — a client would read the truncated text as the
    whole answer. The pieces that did arrive stay in the item: they were handed out.

    The SDK judges both halves: it delivers the failure as a typed frame carrying the reason,
    and it refuses to invent a final response out of a stream that has none."""
    with client.responses.stream(model=BROKEN, input="Where is Paris?") as stream:
        seen = list(stream)
        with pytest.raises(RuntimeError, match=r"response\.completed"):
            stream.get_final_response()

    assert "response.completed" not in [event.type for event in seen]
    failed = seen[-1]
    assert failed.type == "response.failed"
    assert failed.response.status == "failed"
    failure = failed.response.error
    assert failure is not None and "the model fell over" in failure.message
    item = failed.response.output[0]
    assert item.type == "message"
    part = item.content[0]
    assert part.type == "output_text"
    assert part.text == "Par", "the deltas already handed out left the item"


ASKED = "Weather in Paris?"


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


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(client: OpenAI) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function and
    answers with a call beside its text, the result of that call goes back in as items of the
    input, and the second answer is one only a model that was handed the result can give —
    `Script` reads it out of the turn the template rendered.

    Both prompts are compared against the render of the conversation `chat/completions` builds
    out of the same round: the tools nested under `function`, the call on the assistant's turn,
    the result as a turn of its own. Same characters, same ids into the model — which is what
    keeps one checkpoint answering three dialects the same way.
    """
    first = client.responses.create(model=CALLER, input=ASKED, tools=TOOLS)

    assert first.output_text == PREAMBLE
    called = only_call([item for item in first.output if item.type == "function_call"])
    call_id, arguments = called.call_id, called.arguments
    assert json.loads(arguments) == {"city": "Paris"}
    assert last().prompt == rendered(({"role": "user", "content": ASKED},), ENTRIES)

    items: ResponseInputParam = [
        {"role": "user", "content": ASKED},
        {
            "type": "function_call",
            "call_id": call_id,
            "name": "get_weather",
            "arguments": arguments,
        },
        {"type": "function_call_output", "call_id": call_id, "output": RESULT},
    ]
    second = client.responses.create(model=CALLER, input=items, tools=TOOLS)

    assert second.output_text == ANSWERED
    assert [item.type for item in second.output] == ["message"]
    replayed: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": call_id},
    )
    assert last().prompt == rendered(replayed, ENTRIES)


def test_two_calls_replayed_fold_into_the_one_turn_that_made_them(client: OpenAI) -> None:
    """A call is an item of its own here and a key of the assistant's message everywhere else,
    so two of them in a row are one turn and not two — two would tell the model it answered
    twice, and a template that numbers the turns would render a conversation that never
    happened."""
    items: ResponseInputParam = [
        {"role": "user", "content": ASKED},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call", "call_id": "call_2", "name": "get_time", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": RESULT},
        {"type": "function_call_output", "call_id": "call_2", "output": "noon"},
    ]
    client.responses.create(model=SCRIPTED, input=items)

    folded: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": {}},
                },
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "call_1"},
        {"role": "tool", "content": "noon", "tool_call_id": "call_2"},
    )
    assert last().prompt == rendered(folded)


def test_the_sdk_accumulates_the_call_out_of_the_named_frames(client: OpenAI) -> None:
    """What judges the frames is the SDK's own accumulator: an arguments delta for an item it
    was never told about raises there instead of folding in, and `response.completed` is what
    it turns into a final response.

    One delta and the arguments whole inside it — A11's decision — which is the assertion the
    accumulator cannot make: it would concatenate any number of fragments into the same
    string.
    """
    with client.responses.stream(model=CALLER, input=ASKED, tools=TOOLS) as stream:
        seen = list(stream)
        final = stream.get_final_response()

    kinds = [event.type for event in seen]
    assert kinds[-5:] == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    fragments = [
        event.delta for event in seen if event.type == "response.function_call_arguments.delta"
    ]
    assert fragments == [ARGUMENTS], "the arguments arrived in fragments"
    assert [event.sequence_number for event in seen] == list(range(len(seen)))
    assert final.output_text == PREAMBLE
    assert [item.type for item in final.output] == ["message", "function_call"]
    called = only_call([item for item in final.output if item.type == "function_call"])
    assert called.arguments == ARGUMENTS
    assert final.status == "completed"


def test_a_turn_that_only_called_something_carries_no_message_item(client: OpenAI) -> None:
    """The empty message this dialect does not write: an item announced with no text in it is
    an assistant that answered `""` before it called, and a client rendering the transcript
    shows a blank turn. The stream says the same thing — no text frames at all — which is what
    says the item is opened on the first text there is and not before."""
    answer = client.responses.create(model=MUTE, input=ASKED, tools=TOOLS)

    assert [item.type for item in answer.output] == ["function_call"]
    assert answer.output_text == ""

    with client.responses.stream(model=MUTE, input=ASKED, tools=TOOLS) as stream:
        kinds = [event.type for event in stream]
    assert not [kind for kind in kinds if kind.startswith("response.output_text")]
    assert not [kind for kind in kinds if kind.startswith("response.content_part")]
    assert kinds[1] == "response.output_item.added"


def test_tool_choice_none_neither_offers_the_tools_nor_reads_a_call_back(
    client: OpenAI,
) -> None:
    """`none` is honoured where it can be honoured: the tools never reach the prompt, so there
    is nothing to call rather than an instruction not to. And a turn nobody was offered a tool
    for is text — answering with a call would be answering with the one thing the client asked
    us not to do."""
    answer = client.responses.create(model=MUTE, input=ASKED, tools=TOOLS, tool_choice="none")

    assert answer.output_text == ENVELOPE
    assert [item.type for item in answer.output] == ["message"]
    assert last().prompt == rendered(({"role": "user", "content": ASKED},))


def test_strict_is_refused_by_name(client: OpenAI) -> None:
    """`strict` on a *tool* constrains the arguments of a call, and what a grammar constrains
    here is the whole answer (`text.format`): an argument that violated the schema would come
    back as one that was checked against it. The field is declared so that saying so is a named
    error and not the generic refusal an undeclared one would get — and it has to be declared,
    because the SDK's own tool type requires the key on every tool. `false` answers, or the
    refusal would be unreachable."""
    strict: list[FunctionToolParam] = [
        {
            "type": "function",
            "name": "get_weather",
            "description": DESCRIPTION,
            "parameters": SCHEMA,
            "strict": True,
        }
    ]
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=SCRIPTED, input=ASKED, tools=strict)

    assert code(raised.value.body) == "strict_unsupported"
    answer = client.responses.create(model=MUTE, input=ASKED, tools=TOOLS)
    assert [item.type for item in answer.output] == ["function_call"]


def test_a_checkpoint_whose_envelope_nothing_here_parses_answers_with_the_text(
    client: OpenAI,
) -> None:
    """Which family a checkpoint speaks is a fact of its chat template and not of the text it
    writes: read off the output instead, Qwen3.6's `<tool_call><function=…>` is taken for Qwen's
    JSON and the envelope is held for a parser that cannot read it. A template that spells none
    leaves the channel shut, and what the model wrote reaches the client whole."""
    answer = client.responses.create(model=STRANGER, input=ASKED, tools=TOOLS)

    assert answer.output_text == PREAMBLE + ENVELOPE
    assert [item.type for item in answer.output] == ["message"]


def test_an_envelope_the_budget_cut_in_half_comes_back_as_the_text_it_is(client: OpenAI) -> None:
    """Held as a possible call and it is not one: what the model wrote goes out as text. The
    silent failure this rules out is the opposite — an envelope suppressed and no call produced
    reaches the client as a model that chose to call nothing, which is exactly the shape of a
    correct refusal."""
    answer = client.responses.create(model=CUT, input=ASKED, tools=TOOLS)

    assert answer.output_text == '<tool_call>\n{"name": "get_weat'
    assert [item.type for item in answer.output] == ["message"]


def test_the_stream_is_kept_warm_through_a_prefill_longer_than_the_tick(stand: Stand) -> None:
    """Read raw, because what is asserted is a line the SDK is required to ignore: a comment
    frame between the opening events and the first token. Without it a client whose read
    timeout is shorter than the prefill drops the connection before the answer starts — and a
    stream that merely *looked* right would still parse, which is why the SDK cannot judge
    this one."""
    body = {"model": SLOW, "input": "Where is Paris?", "stream": True}
    with (
        httpx.Client() as http,
        http.stream(
            "POST", f"{stand.base_url}/api/openai/v1/responses", json=body, timeout=30
        ) as response,
    ):
        assert response.status_code == 200
        lines = list(response.iter_lines())

    comments = [index for index, line in enumerate(lines) if line.startswith(":")]
    deltas = [
        index for index, line in enumerate(lines) if line == "event: response.output_text.delta"
    ]
    assert comments, "the stream went silent through the whole prefill"
    assert deltas, "the stream never carried a token"
    assert comments[0] < deltas[0], "the keep-alive arrived after the answer already had"


def test_a_generation_the_budget_cut_is_incomplete_and_says_why(client: OpenAI) -> None:
    """`completed` over a sentence `max_output_tokens` cut is a truncation reported as the
    final answer. The dialect has a status for it and a field that names the reason, and this
    is the only place a client can read either."""
    answer = client.responses.create(model=SCRIPTED, input="Where is Paris?", max_output_tokens=1)

    assert answer.status == "incomplete"
    assert answer.incomplete_details is not None
    assert answer.incomplete_details.reason == "max_output_tokens"


def test_a_generation_that_ended_on_its_own_is_completed(client: OpenAI) -> None:
    """The other half: the same script under a budget it does not reach."""
    answer = client.responses.create(model=SCRIPTED, input="Where is Paris?", max_output_tokens=64)

    assert answer.status == "completed"
    assert answer.incomplete_details is None


def test_the_text_and_the_call_of_one_generation_replay_as_one_turn(client: OpenAI) -> None:
    """The canonical loop of this dialect: the client sends back `input + response.output`,
    and a generation that wrote text *and* called something is two items — a message and a
    function_call. They are one turn of the model, and two assistant turns in the prompt tell
    it that it answered twice."""
    items: ResponseInputParam = [
        {"role": "user", "content": ASKED},
        # The item as the dialect wrote it, `id` and `status` included: what a client
        # replays is `input + response.output`, and those two are part of what it got back.
        {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Let me check.", "annotations": []}],
        },
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": RESULT},
    ]
    client.responses.create(model=SCRIPTED, input=items)

    folded: tuple[Turn, ...] = (
        {"role": "user", "content": ASKED},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {}},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "call_1"},
    )
    assert last().prompt == rendered(folded)


def test_a_checked_answer_is_the_document_and_not_the_text_around_it(client: OpenAI) -> None:
    """Level 1 end to end in this dialect: the schema enters the prompt as its last turn and
    what comes back is the value that was checked.

    The prose and the fence stay out of the answer. A client that asked for a schema and got
    `Sure! ```json…``` ` back was handed something `json.loads` refuses, and the document is
    the one thing this route can promise: it is what it validated. The instruction is the
    engine's own `json_instruction` and not a second copy — what is asked for and what
    `validate` enforces cannot be two readings of the same vocabulary.
    """
    answer = client.responses.create(model=WRITER, input=ASKED, text=CHECKED)

    assert json.loads(answer.output_text) == DOCUMENT
    assert "Sure!" not in answer.output_text
    turns: tuple[ChatMessage, ...] = (
        {"role": "user", "content": ASKED},
        {"role": "system", "content": json_instruction(SCHEMA)},
    )
    assert last().prompt == rendered(turns)


def test_a_format_of_text_is_the_default_and_asks_for_nothing(client: OpenAI) -> None:
    """The third member of the union, and what keeps the test above from being an assertion
    that every request gets an instruction: `{"type": "text"}` is what a client that wants
    prose sends, and a route that read it as "some format was named" would put a JSON
    instruction in the prompt and refuse the answer that came back."""
    plain: ResponseTextConfigParam = {"format": {"type": "text"}}
    answer = client.responses.create(model=WRITER, input=ASKED, text=plain)

    assert answer.output_text == "".join(WRITTEN)
    assert last().prompt == rendered(({"role": "user", "content": ASKED},))


def test_a_document_that_breaks_the_schema_is_a_named_error_and_never_a_success(
    client: OpenAI,
) -> None:
    """The failure this level exists to make visible: JSON that parses, a document that does
    not conform, and a 200 with it inside. The path is in the message because a client told
    "invalid" and not *where* is left diffing document against schema by hand.

    422 and not a 5xx: the SDKs retry a 5xx by themselves, and a whole generation bought
    behind the client's back is exactly the cost this level is supposed to show."""
    with pytest.raises(UnprocessableEntityError) as raised:
        client.responses.create(model=BREAKER, input=ASKED, text=CHECKED)

    body = entry(raised.value.body)
    assert code(body) == "schema_violation"
    message = body["message"]
    assert isinstance(message, str)
    assert "$.city is required and missing" in message
    assert "after 1 generation" in message, "what it cost is part of the answer"


def test_an_answer_with_no_json_in_it_is_the_other_failure_and_says_which(
    client: OpenAI,
) -> None:
    """A model that answered in prose is not a model that broke the schema, and the two are
    not one error: what a client does about them differs, and only one of them has a path.

    `json_object` is the format with no schema under it — all it asks is that the answer be
    JSON, which is the half of level 1 that has nothing to validate against."""
    with pytest.raises(UnprocessableEntityError) as raised:
        client.responses.create(model=SCRIPTED, input=ASKED, text=ONLY_JSON)

    assert code(raised.value.body) == "malformed_json"
    turns: tuple[ChatMessage, ...] = (
        {"role": "user", "content": ASKED},
        {"role": "system", "content": json_instruction(None)},
    )
    assert last().prompt == rendered(turns)


def test_a_stream_checks_what_it_already_sent_and_fails_the_response(client: OpenAI) -> None:
    """A stream has one pass: the frames are gone by the time the document can be checked, so
    the violation travels the way a generation that died travels — `response.failed`, which is
    the one frame the SDK's accumulator refuses to build a final response out of. Closing with
    `response.completed` instead would hand the client a document it believes was checked."""
    with client.responses.stream(model=BREAKER, input=ASKED, text=CHECKED) as stream:
        seen = list(stream)
        with pytest.raises(RuntimeError, match=r"response\.completed"):
            stream.get_final_response()

    assert "response.completed" not in [event.type for event in seen]
    last_event = seen[-1]
    assert last_event.type == "response.failed"
    failure = last_event.response.error
    assert failure is not None and "$.city is required and missing" in failure.message
    item = last_event.response.output[0]
    assert item.type == "message"
    part = item.content[0]
    assert part.type == "output_text"
    assert part.text == FAULTY, "what did arrive is still the client's"


def test_a_strict_format_is_decoded_under_the_schema_and_never_asked_for_it(
    stand: Stand, client: OpenAI
) -> None:
    """Level 3's wiring, which is all a stand of scripted models can show: the schema is
    compiled against the model the request named and the walk that comes back is what the
    generation runs under. A route that compiled it and dropped the walk would answer 200 with
    a free decode, which is the guarantee broken silently.

    Nothing goes into the prompt: the mask is the whole of it, and a schema in the prompt as
    well would be paying for the answer twice. And nothing is checked afterwards either — the
    answer is not measured against a schema decoding could not violate.
    """
    answer = client.responses.create(model=GUIDED, input=ASKED, text=GUARANTEED)

    assert json.loads(answer.output_text) == DOCUMENT
    assert stand.engine.compiled[-1] == SCHEMA
    assert last().options.constraint is not None
    assert last().prompt == rendered(({"role": "user", "content": ASKED},))


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scripted model is in no catalog and holds no tokenizer, so there is no token table to
    compile against. What the client must not get is the answer anyway: `strict` asks for a
    guarantee, and a 200 carrying a free decode is that guarantee broken silently.

    The catalog is emptied for the read the engine makes on the way to that refusal: it walks
    the machine's own Hub cache, and what this asserts has nothing to do with what someone
    happened to have downloaded."""
    monkeypatch.setattr(catalog, "scan", list)

    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=WRITER, input=ASKED, text=GUARANTEED)

    assert code(raised.value.body) == "not_constrainable"


def test_a_strict_format_and_tools_are_refused_together(client: OpenAI) -> None:
    """The one combination `strict` makes impossible rather than expensive: the mask allows the
    schema's ids from the first token, so an offered function can never be called. A 200 with
    the tools silently uncallable is the answer this refusal exists instead of.

    `tool_choice: "none"` is not this case — the tools never enter the prompt, so there is
    nothing being dropped."""
    with pytest.raises(BadRequestError) as raised:
        client.responses.create(model=GUIDED, input=ASKED, tools=TOOLS, text=GUARANTEED)

    assert code(raised.value.body) == "strict_with_tools"
    answered = client.responses.create(
        model=GUIDED, input=ASKED, tools=TOOLS, tool_choice="none", text=GUARANTEED
    )
    assert json.loads(answered.output_text) == DOCUMENT


def test_a_turn_that_called_something_is_not_an_answer_to_check(client: OpenAI) -> None:
    """A format is about the answer, and a turn that called a function has not answered yet:
    checking the preamble against the schema would turn every tool call under a format into a
    422, which is refusing the one thing the tools were offered for."""
    answer = client.responses.create(model=CALLER, input=ASKED, tools=TOOLS, text=CHECKED)

    assert [item.type for item in answer.output] == ["message", "function_call"]
    assert answer.output_text == PREAMBLE
