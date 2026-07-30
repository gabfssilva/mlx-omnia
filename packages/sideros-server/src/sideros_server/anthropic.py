"""`/api/anthropic/v1/*` — the dialect Claude Code speaks.

Three things separate it from OpenAI's, and each one is a decision this module owns.

The system prompt is a **field of the request**, not a turn with a role. What the engine
takes is a `Chat`, and what a checkpoint's template renders is turns — so the translation
is explicit here, and it is the one thing in the module that has no counterpart upstream.

The stream is **named events**. The SDK's decoder dispatches on the `event:` line and drops
a frame that carries none, so the name is not decoration around an opaque `data:` the way
it is in OpenAI's; a frame whose name is missing is a frame the client never sees.

The error envelope is its own — `{"type": "error", "error": {...}}`. The SDK picks the
exception class by **status** and hands the body over raw, so what `type` says inside it is
what a client's own error mapping (litellm's, Claude Code's) reads. `encode_error` is the
whole of it, and it is also what `app`'s validation handler calls for a body under this
prefix, so a refused body comes out in this shape instead of FastAPI's `{"detail": [...]}`.

Tools are where the block list earns itself. A call is a `tool_use` block of the assistant's
own message and a result is a `tool_result` block of the *user's* — one message carries every
result of a round — while the conversation the engine takes has one turn per result. So a
message becomes turns here, not a turn, and the results come out before the text that
travelled with them.

Structured output has **one** spelling here and it is a guarantee: `output_config.format`,
which carries a JSON schema and no flag to soften it — upstream the answer is decoded under
that schema, not checked against it afterwards. So this route has no level 1 at all: a schema
either compiles into a grammar or the request is refused in the compiler's own words, which is
the same answer the client would get upstream for a schema outside the supported subset. What
this dialect had before the field existed — forcing a tool and reading its arguments — stays
refused by name (`tool_choice: {"type": "any"}` and `{"type": "tool"}`), for the reason it
always was: forcing a call is a constraint on decoding that this server does not implement,
and answering `auto` to a client that asked for one is a call the model may never have made.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from itertools import count
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from sideros import (
    Chat,
    GenerationOptions,
    ImagePart,
    LogitFilter,
    TextPart,
    UnsupportedInput,
    greedy,
    min_p,
    repetition_penalty,
    sampler,
    temperature,
    top_k,
    top_p,
)
from sideros.chat import tool_family_of
from sideros.generate import Constraint
from sideros.grammar import GrammarRefused
from sideros.suppress import Segment
from sideros.tools import ToolCall
from sideros_server import profiles
from sideros_server.engine import Engine, Job, NotConstrainable
from sideros_server.profiles import Sampling, StoreDep
from sideros_server.responses import (
    Calls,
    ToolTurn,
    UnreadableImage,
    content_of,
    declared,
    image_part,
    unsupported_reason,
)

_KEEP_ALIVE_SECONDS = 0.5


class TextBlock(BaseModel):
    """Unknown keys are dropped rather than refused. A block carries hints that say nothing
    about what the model is asked to write — `cache_control` is the one every Claude Code
    request puts on its system prompt — and refusing the prompt over one leaves the client
    with no way to send the prompt at all. A block of another `type` still fails: a `document`
    silently dropped would be a request answered about something else."""

    type: Literal["text"]
    text: str


class ImageSource(BaseModel):
    """The bytes themselves, and only those: `{"type": "url"}` and `{"type": "file"}` are this
    dialect's other two sources, and each names an image somebody else holds — one would have
    the daemon fetching at a client's word, the other names a file this server never stored.
    `media_type` is narrowed to the one format read here so that a jpeg is refused by name,
    with the name of the field that was wrong."""

    type: Literal["base64"]
    media_type: Literal["image/png"]
    data: str


class ImageBlock(BaseModel):
    """Unknown keys are dropped for `TextBlock`'s reason: `cache_control` rides these too."""

    type: Literal["image"]
    source: ImageSource


class ToolUseBlock(BaseModel):
    """The call the model made, replayed by the client on the next turn. Unknown keys are
    dropped for `TextBlock`'s reason: `cache_control` rides these too."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, object]


class ToolResultBlock(BaseModel):
    """What the client's own function returned, in the user message that answers the round.

    `is_error` is dropped rather than refused, and it is the one drop in this module that
    loses something: no chat template in circulation has a place for it, so what says the
    call failed is the text of the result, which is what a client writes there anyway.
    """

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[TextBlock] = ""


type Block = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    """No `system`: in this dialect it is a field of the request, not a role."""
    content: str | list[Annotated[Block, Field(discriminator="type")]]
    """The discriminator is what makes a refusal worth reading: without it a block with one
    field wrong fails once per member of the union, and what the client is told about is
    whichever member pydantic tried first."""


class Tool(BaseModel):
    """Unknown keys are dropped, `TextBlock`'s reason again — `cache_control` sits on the
    last tool of every Claude Code request. A built-in tool (`{"type": "bash_20250124"}`)
    carries no `input_schema` and is refused by that: this server executes nothing, and a
    definition the model was never shown is a tool it cannot call."""

    name: str
    description: str | None = None
    input_schema: dict[str, object]


class ToolChoice(BaseModel):
    """`any` and `tool` are refused by name: forcing a call is a constraint on decoding, and
    answering `auto` to a client that asked for one is a call the model may never have made.
    `disable_parallel_tool_use` goes with them — nothing here decides how many calls a
    generation writes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["auto", "none"]


class JsonFormat(BaseModel):
    """The one output format the dialect spells. The schema arrives under `schema`, which is
    an alias because a pydantic field of that name shadows `BaseModel.schema`; it is required
    here because it is required there — a format with nothing to conform to is not one."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    definition: dict[str, object] = Field(alias="schema")


class OutputConfig(BaseModel):
    """`format` is the guarantee (see the module docstring). `effort` is the other key this
    object carries upstream and is refused by name: it decides how much the model spends on
    an answer, and there is no such dial under this server."""

    model_config = ConfigDict(extra="forbid")

    format: JsonFormat | None = None


class MessagesRequest(BaseModel):
    """Unknown fields are refused rather than dropped, `stop_sequences` among them: the
    engine's `stop` is a set of token ids and cannot express a string, so a client told its
    sequence was honoured would read a truncation that never happened."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[Message]
    max_tokens: int = Field(gt=0)
    """Required, which is the dialect's own decision and not an omission: Anthropic has no
    default budget, and every SDK sends one."""
    system: str | list[TextBlock] | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    output_config: OutputConfig | None = None
    """The schema this answer is decoded under, when the client sent one. There is no checked
    flavour of it in this dialect and none is invented here: the schema is compiled into a
    grammar and the ids that would break it are at -inf before the draw, or the request is
    refused with the compiler's own words."""
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    """Upstream's range and upstream's default: the answer to a request that names no
    temperature is drawn, not argmaxed."""
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)


def encode_error(status: int, kind: str, message: str) -> JSONResponse:
    """The dialect's envelope. `kind` is Anthropic's own vocabulary — `invalid_request_error`,
    `not_found_error`, `api_error` — and travels beside the status rather than instead of
    it, because the SDK reads one and the client's error mapping reads the other."""
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": kind, "message": message}},
    )


def _text(content: str | list[TextBlock]) -> str:
    return content if isinstance(content, str) else "".join(block.text for block in content)


def _called(block: ToolUseBlock) -> dict[str, object]:
    """The call in the shape the templates read: `function.name` and arguments as JSON text,
    which is `chat/completions`'s spelling and the one transformers documents."""
    return {
        "id": block.id,
        "type": "function",
        "function": {"name": block.name, "arguments": json.dumps(block.input)},
    }


def _part(block: TextBlock | ImageBlock) -> TextPart | ImagePart:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return image_part(block.source.data, block.source.media_type)


def _turns(message: Message) -> list[ToolTurn]:
    """One message, and the turns it spells. Its results come first because they are the
    round the message answers, and the text after them is the client's next word. The text and
    image blocks keep the order they arrived in — where an image sits among the words is what
    the template writes a marker for, and what the model then looks at."""
    content = message.content
    if isinstance(content, str):
        return [{"role": message.role, "content": content}]
    said = content_of(
        [_part(block) for block in content if isinstance(block, TextBlock | ImageBlock)]
    )
    made = [_called(block) for block in content if isinstance(block, ToolUseBlock)]
    results = [block for block in content if isinstance(block, ToolResultBlock)]
    turns: list[ToolTurn] = [
        {"role": "tool", "content": _text(block.content), "tool_call_id": block.tool_use_id}
        for block in results
    ]
    if said or made or not results:
        turn: ToolTurn = {"role": message.role, "content": said}
        if made:
            turn["tool_calls"] = made
        turns.append(turn)
    return turns


def _tools(request: MessagesRequest) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: {"type": "none"}` is honoured where it can be honoured: the tools never
    enter the prompt, so the model has nothing to call rather than an instruction not to."""
    choice = request.tool_choice
    if request.tools is None or (choice is not None and choice.type == "none"):
        return ()
    return tuple(
        declared(tool.name, tool.description, tool.input_schema) for tool in request.tools
    )


def _conversation(request: MessagesRequest, preset: str | None) -> Chat:
    """The translation this dialect exists for: `system` is a field on the way in and the
    first turn on the way out. The profile's prompt fills it only when the request left it
    out — the same precedence the sampling knobs below follow."""
    system = _text(request.system) if request.system is not None else preset
    turns: list[ToolTurn] = [] if system is None else [{"role": "system", "content": system}]
    for message in request.messages:
        turns += _turns(message)
    return Chat(tuple(turns), tools=_tools(request))


def _options(
    request: MessagesRequest, sampling: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """The profile fills the knobs the client left out, and only those: a request that names
    a temperature means it. Which ones it left out is `model_fields_set` — the dialect's
    defaults are values like any other, so an unset field cannot be told from an explicit
    one by its value. `min_p`, `repetition_penalty` and `seed` have no field here at all,
    so a profile is the only thing that can set them.

    The constraint composes with all of them and is nobody's filter: the mask is applied
    before the sampler runs, so what is drawn is drawn from what the grammar left."""
    asked = request.model_fields_set
    heat = (
        request.temperature
        if "temperature" in asked or sampling.temperature is None
        else sampling.temperature
    )
    nucleus = request.top_p if "top_p" in asked or sampling.top_p is None else sampling.top_p
    kept = request.top_k if "top_k" in asked or sampling.top_k is None else sampling.top_k
    repeats = sampling.repetition_penalty
    penalty = None if repeats is None else repetition_penalty(repeats)
    if heat == 0.0:
        # The deterministic end of the dial: no distribution is left to draw from, and
        # dividing by it would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=request.max_tokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
        )

    filters: list[LogitFilter] = [temperature(heat)]
    if kept is not None:
        filters.append(top_k(kept))
    if nucleus is not None:
        filters.append(top_p(nucleus))
    if sampling.min_p is not None:
        filters.append(min_p(sampling.min_p))
    return GenerationOptions(
        max_tokens=request.max_tokens,
        sampler=sampler(*filters, seed=sampling.seed),
        penalty=penalty,
        constraint=constraint,
    )


def _message(
    message_id: str,
    model: str,
    content: list[dict[str, object]],
    stop_reason: str | None,
    usage: Mapping[str, int],
) -> dict[str, object]:
    """The answer, and the `message` that opens a stream: the same object, once with the
    content and the reason it ended on and once with neither."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _use(block_id: str, name: str, input: Mapping[str, object]) -> dict[str, object]:
    """A call, as a block of the answer. `input` is empty in the block that opens one on a
    stream: what fills it is the `input_json_delta` that follows."""
    return {"type": "tool_use", "id": block_id, "name": name, "input": input}


def _stop_reason(job: Job, max_tokens: int, called: bool) -> str:
    """The budget, a call, or the model's own end token. `tool_use` outranks the budget: what
    is reported is only ever a call read whole, so a client that gets one can execute it, and
    `max_tokens` beside an executable call sends it to render a truncation instead.
    `stop_sequence` never appears: no request can ask for one (see `MessagesRequest`), so it
    is not a reason anything here can end for."""
    if called:
        return "tool_use"
    return "max_tokens" if job.meter.completion_tokens >= max_tokens else "end_turn"


def _frame(event: str, payload: Mapping[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _delta(index: int, piece: str) -> str:
    return _frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": piece},
        },
    )


async def _pieces(job: Job) -> AsyncIterator[Segment | None]:
    """The generation's pieces, with a `None` for each keep-alive the wait owes: a long
    prefill would otherwise leave the connection silent until the first token, and the
    client drops it."""
    while True:
        try:
            piece = await asyncio.wait_for(job.chunks.get(), _KEEP_ALIVE_SECONDS)
        except TimeoutError:
            yield None
            continue
        if piece is None:
            return
        yield piece


async def _events(
    job: Job, message_id: str, model: str, max_tokens: int, calls: Calls | None
) -> AsyncIterator[str]:
    """`message_start` waits for the first piece instead of going out at once, because it
    carries `input_tokens` and the prompt is only counted once the checkpoint's own template
    has rendered it and the model has tokenized it — which happens on the other side of the
    queue. What holds the connection open until then is the ping the dialect has for it."""
    blocks = count()
    text_index: int | None = None

    def deltas(content: str) -> Iterator[str]:
        """The text block is opened on the first text there is, not before: a generation that
        only called something has no text block, and one opened empty is an assistant that
        answered `""` before it called."""
        nonlocal text_index
        if text_index is None:
            text_index = next(blocks)
            yield _frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": text_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        yield _delta(text_index, content)

    def emitted(piece: Segment) -> Iterator[str]:
        # No family, nothing that could read a channel back: every segment is text the client
        # asked for, whichever one the model wrote it on.
        content = piece.text if calls is None else calls.push(piece)
        if content:
            yield from deltas(content)

    try:
        pieces = _pieces(job)
        first: Segment | None = None
        async for piece in pieces:
            if piece is None:
                yield _frame("ping", {"type": "ping"})
                continue
            first = piece
            break

        usage = {"input_tokens": job.meter.prompt_tokens, "output_tokens": 0}
        yield _frame(
            "message_start",
            {"type": "message_start", "message": _message(message_id, model, [], None, usage)},
        )
        if first is not None:
            for frame in emitted(first):
                yield frame
        async for piece in pieces:
            if piece is None:
                yield _frame("ping", {"type": "ping"})
                continue
            for frame in emitted(piece):
                yield frame

        made: tuple[ToolCall, ...] = ()
        if calls is not None:
            tail, made = calls.finish()
            if tail:
                for frame in deltas(tail):
                    yield frame
        if job.error is not None:
            # The status is long gone — the response opened 200 the moment the first frame
            # went out — so the dialect's own event is the only place left to say it.
            yield _frame(
                "error", {"type": "error", "error": {"type": "api_error", "message": job.error}}
            )
            return
        if text_index is not None:
            yield _frame("content_block_stop", {"type": "content_block_stop", "index": text_index})
        for call in made:
            # The call whole, in the one delta its block carries: the SDK's accumulator
            # concatenates `partial_json` and parses what it has at every step, so a single
            # fragment is the valid degenerate case of that. A11's decision.
            index = next(blocks)
            yield _frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": _use(f"toolu_{uuid.uuid4().hex}", call.name, {}),
                },
            )
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(call.arguments),
                    },
                },
            )
            yield _frame("content_block_stop", {"type": "content_block_stop", "index": index})
        yield _frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _stop_reason(job, max_tokens, bool(made)),
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": job.meter.completion_tokens},
            },
        )
        yield _frame("message_stop", {"type": "message_stop"})
    finally:
        job.cancel()


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


@router.get("/api/anthropic/v1/models")
def models(store: StoreDep) -> dict[str, object]:
    """The catalog, and every name each model answers to — the same window OpenAI's route
    opens, in this dialect's shape. `has_more` is false rather than absent: the SDK pages
    until something says to stop, and a missing flag with a `last_id` is a next page."""
    return {
        "data": [
            {
                "id": served,
                "type": "model",
                "display_name": served,
                "created_at": "1970-01-01T00:00:00Z",
            }
            for served in profiles.served_ids(store)
        ],
        "has_more": False,
    }


def _said(message: Message) -> bool:
    """Whether the client wrote anything in this message. A block list is not judged by its
    text: a turn that only carries the result of a call said what it had to say."""
    content = message.content
    if isinstance(content, str):
        return bool(content)
    return any(block.text if isinstance(block, TextBlock) else True for block in content)


@router.post("/api/anthropic/v1/messages", response_model=None)
async def messages(
    request: MessagesRequest, engine: EngineDep, store: StoreDep
) -> JSONResponse | StreamingResponse:
    if not any(_said(message) for message in request.messages):
        return encode_error(
            400, "invalid_request_error", "messages must contain non-empty content"
        )
    config = request.output_config
    schema = None if config is None or config.format is None else config.format.definition
    if schema is not None and _tools(request):
        return encode_error(
            400,
            "invalid_request_error",
            "output_config.format and tools cannot both be honoured: the grammar constrains "
            "decoding to the schema from the first token, so the model cannot write a call "
            "however it is offered. Send the schema alone, or offer no tools.",
        )

    model_id, profile = profiles.resolve(store, request.model)
    try:
        conversation = _conversation(request, None if profile is None else profile.system_prompt)
    except UnreadableImage as unreadable:
        return encode_error(400, "invalid_request_error", str(unreadable))
    preset = Sampling() if profile is None else profile.sampling
    message_id = f"msg_{uuid.uuid4().hex}"
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        walk = None if schema is None else await engine.constrain(model_id, schema)
        # A name no checkpoint answers to and one whose load fails are the same answer to the
        # client: this model is not available here. A `model:typo` lands here whole — no
        # profile matched, so the name was never split.
        job = await engine.submit(model_id, conversation, _options(request, preset, walk))
    except GrammarRefused as refusal:
        # The compiler's own words — `Unimplemented keys: ["uniqueItems"]` is a reason where
        # "grammar error" is not, and it is what tells the client which keyword to drop.
        return encode_error(400, "invalid_request_error", str(refusal))
    except NotConstrainable as refusal:
        return encode_error(400, "invalid_request_error", str(refusal))
    except UnsupportedInput:
        return encode_error(
            400, "invalid_request_error", unsupported_reason(request.model, conversation)
        )
    except Exception as error:
        return encode_error(
            404, "not_found_error", f"model {request.model!r} is not available: {error}"
        )

    # No tools offered, nothing to read back; no family, nothing that could read it. Both
    # reach the client the way the generation always did, piece for piece: suppressing an
    # envelope no parser answers to costs the client the text and gives back no call.
    family = tool_family_of(job.model) if conversation.tools else None
    calls = None if family is None else Calls(family)

    if request.stream:
        return StreamingResponse(
            _events(job, message_id, request.model, request.max_tokens, calls),
            media_type="text/event-stream",
        )

    pieces: list[Segment] = []
    try:
        while (piece := await job.chunks.get()) is not None:
            pieces.append(piece)
    finally:
        job.cancel()
    if job.error is not None:
        return encode_error(500, "api_error", job.error)
    text = "".join(piece.text for piece in pieces)
    made: tuple[ToolCall, ...] = ()
    if calls is not None:
        held = "".join(calls.push(piece) for piece in pieces)
        tail, made = calls.finish()
        text = held + tail
    # A turn that only called something carries no text block: an empty one is an assistant
    # that answered with nothing, which is not what happened.
    content: list[dict[str, object]] = (
        [] if made and not text else [{"type": "text", "text": text}]
    )
    content += [_use(f"toolu_{uuid.uuid4().hex}", call.name, call.arguments) for call in made]
    return JSONResponse(
        content=_message(
            message_id,
            request.model,
            content,
            _stop_reason(job, request.max_tokens, bool(made)),
            {
                "input_tokens": job.meter.prompt_tokens,
                "output_tokens": job.meter.completion_tokens,
            },
        )
    )
