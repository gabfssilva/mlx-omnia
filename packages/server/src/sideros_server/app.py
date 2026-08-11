import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from importlib.metadata import version
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from sideros import (
    Chat,
    GenerationOptions,
    ImagePart,
    LogitFilter,
    Penalty,
    Sampler,
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
from sideros import ChatMessage as Turn
from sideros.chat import parser_of
from sideros.footprint import ceiling
from sideros.generate import Constraint, Meter
from sideros.grammar import GrammarRefused
from sideros.parsers import Segment, ToolCall, unmarked
from sideros.schema import MalformedJSON, SchemaViolation
from sideros_server import (
    anthropic,
    auth,
    bench,
    benchmarks,
    catalog,
    config,
    datasets,
    download,
    events,
    features,
    fidelity,
    gemini,
    jobs,
    metrics,
    prefixes,
    profiles,
    quantize,
    residency,
    responses,
    sessions,
    state,
    system,
    tokenization,
)
from sideros_server.engine import Engine, Job, NotConstrainable, NotQuantizable
from sideros_server.jobs import Jobs
from sideros_server.profiles import Sampling, StoreDep
from sideros_server.responses import (
    PROFILE_ONLY,
    Calls,
    Checked,
    OpenAIEffort,
    ToolTurn,
    UnreadableImage,
    content_of,
    declared,
    document,
    effort_of,
    failed,
    inline_image,
    instruction,
    openai_envelope,
    openai_error,
    unsupported_reason,
)
from sideros_server.store import Store

_KEEP_ALIVE_SECONDS = 0.5

_STARTED = time.monotonic()
"""When this process came up, by the clock that does not move when the system one does."""

_LIVE_BEAT_SECONDS = 0.5
"""How often the request being served is republished. Fast enough that a decode rate reads
as a rate, slow enough that a screen is not redrawn per token."""


async def _beating(register: metrics.Metrics) -> None:
    """Nothing calls the register while a request runs, so nothing would say how it is
    going. This is the one thing in the daemon on a clock rather than an edge."""
    while True:
        await asyncio.sleep(_LIVE_BEAT_SECONDS)
        register.beat()


class CalledFunction(BaseModel):
    name: str
    arguments: str
    """JSON text, which is how the dialect spells the arguments. It reaches the template
    unparsed: the ones in circulation render a string argument as it stands."""


class CalledTool(BaseModel):
    id: str
    type: Literal["function"]
    function: CalledFunction


class ImageUrl(BaseModel):
    """Where this dialect puts an image. The only URL accepted is the `data:` one that carries
    the bytes: fetching an `https://` one would have the daemon making requests of its own, at
    a client's word, from inside the network it was told to serve.

    `detail` is declared so `low` and `high` are refused by name rather than dropped. Upstream
    they buy more tiles of the image and cost more; here the checkpoint's own processor decides
    the resolution from the image itself, so a client told `high` was honoured was told
    nothing."""

    url: str
    detail: Literal["auto"] = "auto"


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


type Part = TextContent | ImageContent


class ChatMessage(BaseModel):
    """Unknown fields are dropped here rather than refused, unlike the request's own: a
    message is history, and what a client replays is the message the dialect wrote — which
    carries `index` inside a call and a `refusal` next to the content, neither of which
    means anything on the way back."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Annotated[Part, Field(discriminator="type")]] | None = None
    """`None` is the assistant turn that only called something: the call is the message.

    The discriminator is what a refusal is worth reading: without it a part with one field
    wrong fails once per member of the union, and what the client is told about is whichever
    member pydantic tried first."""
    tool_calls: list[CalledTool] | None = None
    tool_call_id: str | None = None
    """Which call this result answers. The role `tool` is what carries it."""


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None
    """The JSON schema, untouched: the template renders the whole entry with `tojson`, so
    what the model reads is what the client wrote."""


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: Function


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class SchemaSpec(BaseModel):
    """The dialect's `json_schema` object. The schema itself arrives under `schema`, which is
    an alias because a pydantic field of that name shadows `BaseModel.schema`.

    `strict` is what tells the two levels apart: with it the schema is compiled into a grammar
    and decoding cannot produce a violation, without it the schema enters the prompt and the
    answer is checked afterwards. A schema the compiler will not take is refused in its own
    words rather than quietly demoted — the client asked for a guarantee."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    definition: dict[str, object] | None = Field(default=None, alias="schema")
    """Absent is `json_object` with a name on it: there is no schema to check against, so what
    is asked of the answer is that it be JSON."""
    strict: bool | None = None


class TextFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]


class JsonObjectFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: SchemaSpec


type ResponseFormat = TextFormat | JsonObjectFormat | JsonSchemaFormat


class ChatRequest(BaseModel):
    # Unknown fields are refused rather than dropped: a client that asks for `logit_bias`
    # and gets an answer has been told, wrongly, that it was honoured.
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto"] = "auto"
    """`required` and a named function are refused by name: forcing a call is a constraint
    on decoding, and accepting the field without one answers with a call the model may
    never have made."""
    max_tokens: int = Field(default=128, gt=0)
    reasoning_effort: OpenAIEffort | None = None
    """How hard the checkpoint is asked to think. There is no budget beside it because
    upstream has none: a client that wants one names a profile, which is where the knobs
    this dialect cannot spell already live."""
    stream: bool = False
    # Ignored when `stream` is false, and honestly so: the non-streaming answer carries
    # `usage` either way, so nothing the client asked for goes unanswered.
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = None
    response_format: Annotated[ResponseFormat, Field(discriminator="type")] | None = None
    """Which of the two levels this answer gets. Without `strict` the schema enters the prompt
    as a turn and the answer is checked against it once the generation is spent, which is a
    check that can fail; with it the schema is compiled into a grammar and the mask makes the
    violation unreachable, at the price of a sync per step.

    The discriminator is what makes a refusal worth reading: without it a body with one field
    wrong fails once per member of the union, and the client is told about whichever member
    pydantic tried first."""
    max_schema_attempts: int = Field(default=1, ge=1, le=4)
    """How many whole generations the client will pay for to get an answer that validates.

    One by default, and the number is in every structured answer: each attempt is an entire
    generation on a queue every request shares, and a retry the client cannot see is a model
    that cannot obey the schema, hidden behind a bill nobody reads. The ceiling is that same
    queue — four generations is the longest one request may hold it."""


def _options(
    request: ChatRequest, sampling: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """The dialect's defaults are OpenAI's, so an unset `temperature` is 1.0 and the answer
    is drawn, not argmaxed. Filters run in the order below — the cuts read the distribution
    temperature already shaped, which is what makes `top_p` mean the same here as upstream.

    The constraint composes with all of them and is nobody's filter: the mask is applied
    before the sampler runs, so what is drawn is drawn from what the grammar left.

    `sampling` is here for the knobs this dialect has no field for — `reasoning_budget` is
    the only one so far — which `_preset` cannot fill because there is nothing to fill."""
    repeats = request.repetition_penalty
    penalty: Penalty | None = None if repeats == 1.0 else repetition_penalty(repeats)
    limit = catalog.context_of(request.model)
    thinking = sampling.reasoning_budget
    if request.temperature == 0.0:
        # The deterministic end of the dial: no distribution is left to draw from, and
        # dividing by it would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=request.max_tokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=thinking,
            context_limit=limit,
        )

    filters: list[LogitFilter] = [temperature(request.temperature)]
    if request.top_k is not None:
        filters.append(top_k(request.top_k))
    if request.top_p < 1.0:
        filters.append(top_p(request.top_p))
    if request.min_p > 0.0:
        filters.append(min_p(request.min_p))
    drawn: Sampler = sampler(*filters, seed=request.seed)
    return GenerationOptions(
        max_tokens=request.max_tokens,
        sampler=drawn,
        penalty=penalty,
        constraint=constraint,
        reasoning_budget=thinking,
        context_limit=limit,
    )


def _preset(request: ChatRequest, sampling: Sampling) -> ChatRequest:
    """The preset — the profile, over the sampling defaults the checkpoint declares — fills
    the knobs the client left out, and only those: a request that names a temperature
    means it, whatever the profile it also named says. Which knobs were
    left out is the request's own `model_fields_set` — the dialect's defaults are values
    like any other, so an unset field cannot be told from an explicit one by its value."""
    filled = {
        knob: value
        for knob, value in sampling.model_dump(exclude_none=True).items()
        if knob not in request.model_fields_set and knob not in PROFILE_ONLY
    }
    return request.model_copy(update=filled)


# `model_copy(update=...)` writes the keys straight into the instance: no validation, and the
# `extra="forbid"` above never sees them. A knob the profile grows and the dialect does not have
# would be set on the request and read by nobody — silently, and only for requests that name a
# profile, which is the one path with no dialect-level schema to catch it. `PROFILE_ONLY` is
# the exception with a reader: those are excluded above and read in `_options`.
assert set(Sampling.model_fields) - PROFILE_ONLY <= set(ChatRequest.model_fields)


def _messages(request: ChatRequest, system_prompt: str | None) -> list[ChatMessage]:
    """The profile's system prompt is the conversation's first turn, unless the client sent
    a system message of its own — same precedence the knobs above follow, and it keeps the
    template from rendering two system turns for it to pick between."""
    if system_prompt is None or any(message.role == "system" for message in request.messages):
        return request.messages
    return [ChatMessage(role="system", content=system_prompt), *request.messages]


def _part(part: Part) -> TextPart | ImagePart:
    return (
        {"type": "text", "text": part.text}
        if isinstance(part, TextContent)
        else inline_image(part.image_url.url)
    )


def _content(content: str | list[Part] | None) -> str | tuple[TextPart | ImagePart, ...]:
    """What the turn carries, in the shape the template reads: characters when the message is
    text and the parts themselves when one of them is an image. `None` is `''` — the assistant
    turn that only called something, whose content the template concatenates."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return content_of([_part(part) for part in content])


def _turn(message: ChatMessage) -> Turn:
    """A call travels in the shape it arrived in — the templates in circulation read
    `function.name` and `function.arguments`, and render the arguments as they stand when
    they are text. The content of a turn that only called something is `''` and not `None`:
    the template concatenates it."""
    turn: ToolTurn = {"role": message.role, "content": _content(message.content)}
    if message.tool_calls is not None:
        turn["tool_calls"] = [call.model_dump() for call in message.tool_calls]
    if message.tool_call_id is not None:
        turn["tool_call_id"] = message.tool_call_id
    return turn


def _tools(request: ChatRequest) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: "none"` is honoured where it can be honoured: the tools never enter the
    prompt, so the model has nothing to call rather than an instruction not to.

    `responses.declared` and not this dialect's own `model_dump`: a template that renders the
    entry with `tojson` writes the keys in the order they are in, so the same function
    declared through two dialects has to be built in one place or the two prompts differ."""
    if request.tools is None or request.tool_choice == "none":
        return ()
    return tuple(
        declared(tool.function.name, tool.function.description, tool.function.parameters)
        for tool in request.tools
    )


def _checked(request: ChatRequest) -> Checked | None:
    """What this request asks to be checked, or `None` when it asks for nothing — `text` is
    the dialect's own default and says exactly that."""
    wanted = request.response_format
    if wanted is None or isinstance(wanted, TextFormat):
        return None
    schema = wanted.json_schema.definition if isinstance(wanted, JsonSchemaFormat) else None
    return Checked(schema, request.max_schema_attempts)


def _strict(request: ChatRequest) -> Mapping[str, object] | None:
    """The schema this request asks to be *guaranteed* rather than checked, or `None` for
    every other shape. Under it decoding is constrained, so there is no answer that violates
    the schema and no attempt to buy; without it the schema is a turn in the prompt and the
    answer is measured against it once the generation is spent."""
    wanted = request.response_format
    if not isinstance(wanted, JsonSchemaFormat) or not wanted.json_schema.strict:
        return None
    return wanted.json_schema.definition


def _refused(request: ChatRequest) -> JSONResponse | None:
    """The four shapes of `response_format` this route cannot honour, each named.

    Both halves of the first two are `strict`'s. Without a schema it promises a decode that
    cannot violate one that was never sent, and reading it as `json_object` answers a demand
    for a guarantee with a demand for JSON. With tools it promises two things at once: the
    grammar constrains decoding to the schema from the first token, so the model cannot write
    a call whatever it was offered, and a 200 with no calls in it is that told as a success.
    """
    wanted = request.response_format
    if (
        isinstance(wanted, JsonSchemaFormat)
        and wanted.json_schema.strict
        and wanted.json_schema.definition is None
    ):
        return openai_error(
            400,
            "strict promises a decode that cannot violate the schema, and this request carries "
            "no schema: json_schema holds it under `schema`, and it is what decoding is "
            "constrained by.",
            "strict_without_schema",
        )
    if _strict(request) is not None and _tools(request):
        return openai_error(
            400,
            "strict and tools cannot both be honoured: the grammar constrains decoding to the "
            "schema from the first token, so the model cannot write a call however it is "
            "offered. Send the same json_schema without strict, or offer no tools.",
            "strict_with_tools",
        )
    if _checked(request) is None:
        if "max_schema_attempts" in request.model_fields_set:
            return openai_error(
                400,
                "max_schema_attempts counts the generations spent making an answer validate, "
                "and this request asks for nothing to be validated: it needs a response_format.",
                "max_schema_attempts",
            )
        return None
    if request.stream and request.max_schema_attempts > 1:
        return openai_error(
            400,
            "max_schema_attempts above 1 cannot be honoured with stream: a second attempt is a "
            "second generation, and the frames of the first one have already left.",
            "stream_attempts",
        )
    return None


def _correction(output: str, failure: MalformedJSON | SchemaViolation) -> tuple[Turn, Turn]:
    """What the next attempt reads: what the model wrote, and what was wrong with it.

    A retry over the same turns is the same generation — greedy decoding lands on the same
    tokens, and a drawn one is a coin flip nobody asked for — so what makes the second attempt
    a second attempt is these two turns. Naming the field is why `SchemaViolation` carries a
    path: a correction that cannot say where it broke invites the same mistake."""
    reason = (
        f"{failure.path} {failure.reason}"
        if isinstance(failure, SchemaViolation)
        else "it carried no JSON value"
    )
    return (
        {"role": "assistant", "content": output},
        {
            "role": "user",
            "content": f"That answer was not accepted: {reason}. Answer again, corrected, "
            "with the JSON value and nothing else.",
        },
    )


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


def _tool_call(index: int, call: ToolCall) -> dict[str, object]:
    """`index` on every entry of a list is the hard part of the shape: the SDK's own
    accumulator matches the entries of two frames by it, and raises on one that has none."""
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }


def _chunk(
    request_id: str, created: int, model: str, delta: Mapping[str, object], finish: str | None
) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _counts(meter: Meter) -> dict[str, int]:
    """The turn in flat integers, which is the shape a retry loop can add up."""
    return {
        "prompt_tokens": meter.prompt_tokens,
        "cached_tokens": meter.reused_tokens,
        "completion_tokens": meter.completion_tokens,
        "total_tokens": meter.prompt_tokens + meter.completion_tokens,
    }


def _usage(counts: Mapping[str, int]) -> dict[str, object]:
    """`prompt_tokens` is the whole prompt, reused or not: it is what the request sent, and
    a client that priced the turn on it would price it the same either way. What the reuse
    changes rides under `prompt_tokens_details`, where the dialect puts it — and it is
    written even when it is zero, so that a client can tell a miss from a server that does
    not carry the field."""
    return {
        "prompt_tokens": counts["prompt_tokens"],
        "prompt_tokens_details": {"cached_tokens": counts["cached_tokens"]},
        "completion_tokens": counts["completion_tokens"],
        "total_tokens": counts["total_tokens"],
    }


def _timings(job: Job) -> dict[str, object]:
    """What the turn cost in seconds, beside what it cost in tokens. The dialect has no field
    for any of it, so it rides under a prefixed key an SDK that does not know it drops — the
    same numbers `/admin/metrics` publishes, so a client reading the stream and a dashboard
    watching the register never disagree.
    """
    meter = job.meter
    rate = meter.tokens_per_second
    per_token = None if job.lease is None else job.lease.active_bytes
    speculation = meter.speculation
    return {
        "load_seconds": job.load_seconds,
        # Absent for a turn that did not speculate, which is every model with no drafter and
        # every request a drafter could not verify — a sampled one, a grammar. Present, it is
        # the one place the difference shows: the rate alone never says a drafter was there.
        "speculation": (
            None
            if speculation.rounds == 0
            else {
                "rounds": speculation.rounds,
                "proposed": speculation.proposed,
                "accepted": speculation.accepted,
            }
        ),
        "ttft_seconds": meter.ttft,
        "prefill_tokens_per_second": metrics.prefill_rate(
            meter.prompt_tokens, meter.ttft, meter.reused_tokens
        ),
        "tokens_per_second": rate,
        "bytes_per_token": per_token,
        "ceiling_fraction": (
            None if rate is None or per_token is None else rate / ceiling(per_token)
        ),
    }


def _usage_chunk(request_id: str, created: int, model: str, job: Job) -> str:
    """The extra frame `stream_options.include_usage` asks for: the whole request's usage
    and no choices, which is the shape the SDK folds into the completion it accumulates."""
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": _usage(_counts(job.meter)),
        "x_sideros": _timings(job),
    }
    return f"data: {json.dumps(payload)}\n\n"


def _finish(job: Job, max_tokens: int, called: bool) -> str:
    """Which of the three ended the turn. `tool_calls` outranks the budget for the reason the
    Anthropic dialect gives: only a call read whole is ever reported, so a client that gets
    one can execute it, and `length` beside an executable call sends it to render a truncation
    instead. `length` is what an agent loop branches on to continue — reporting `stop` for a
    generation the budget cut hands it a half sentence as the final answer."""
    if called:
        return "tool_calls"
    budget = max_tokens
    context = catalog.context_of(job.model_id)
    if context is not None:
        # The generation ran under the window's own cap (see `GenerationOptions.
        # context_limit`): a turn that cap cut is `length`, whatever the client asked for.
        budget = min(budget, max(0, context - job.meter.prompt_tokens))
    return "length" if job.meter.completion_tokens >= budget else "stop"


async def _drain(job: Job) -> list[Segment]:
    pieces: list[Segment] = []
    while (piece := await job.chunks.get()) is not None:
        pieces.append(piece)
    return pieces


async def _events(
    job: Job,
    request_id: str,
    created: int,
    model: str,
    usage: bool,
    calls: Calls | None,
    max_tokens: int,
    checked: Checked | None,
) -> AsyncIterator[str]:
    sent: list[str] = []
    try:
        yield _chunk(request_id, created, model, {"role": "assistant", "content": ""}, None)
        while True:
            try:
                piece = await asyncio.wait_for(job.chunks.get(), _KEEP_ALIVE_SECONDS)
            except TimeoutError:
                # Keeps the connection warm through a long prefill.
                yield ": keep-alive\n\n"
                continue
            if piece is None:
                break
            if piece.channel == "reasoning":
                # The field the dialect has for it, which is what DeepSeek, vLLM and Ollama
                # answer with over this same route. Without it the block goes out as the
                # answer: a client draws `<think>` and everything under it as what the model
                # said, and the first token of the answer arrives hundreds of tokens late.
                if thought := unmarked(piece.text):
                    yield _chunk(request_id, created, model, {"reasoning_content": thought}, None)
                continue
            # No family, nothing that could read a channel back: every segment is text the
            # client asked for, whichever one the model wrote it on.
            if content := (piece.text if calls is None else calls.push(piece)):
                sent.append(content)
                yield _chunk(request_id, created, model, {"content": content}, None)
        made: tuple[ToolCall, ...] = ()
        if calls is not None:
            tail, made = calls.finish()
            if tail:
                sent.append(tail)
                yield _chunk(request_id, created, model, {"content": tail}, None)
        if job.error is not None:
            # A frame carrying `error` is what the SDK raises on, and the only way to fail a
            # request that already answered 200. Without it a generation that died — a render
            # the template refused, an allocation that could not be made — closes with
            # `finish_reason: "stop"` and `[DONE]`, which is a completed answer: the client
            # accumulates whatever text arrived before the failure as the whole of it.
            failure = openai_envelope(job.error, "generation_failed", "server_error")
            yield f"data: {json.dumps({'error': failure})}\n\n"
            return
        if checked is not None and not made:
            try:
                document("".join(sent), checked)
            except (MalformedJSON, SchemaViolation) as violation:
                # A stream cannot take back the text it already handed out, and a second
                # attempt is refused for it (`_refused`): what is left is the same door a
                # generation that died goes through, which is the only way to fail a request
                # that already answered 200. The frames stay as they were sent — what failed
                # is the answer, not the client's reading of it.
                reason, code = failed(violation, 1)
                rejected = openai_envelope(reason, code, "server_error")
                yield f"data: {json.dumps({'error': rejected})}\n\n"
                return
        for index, call in enumerate(made):
            # One frame per call, and the call whole inside it: the arguments never arrive
            # in fragments, which is A11's decision. What a client loses is an argument
            # rendered as it is written, not compatibility.
            delta = {"tool_calls": [_tool_call(index, call)]}
            yield _chunk(request_id, created, model, delta, None)
        yield _chunk(request_id, created, model, {}, _finish(job, max_tokens, bool(made)))
        if usage:
            # After the finish frame and before [DONE], where the dialect puts it: the
            # job is over by now, so the meter is the whole request's.
            yield _usage_chunk(request_id, created, model, job)
        yield "data: [DONE]\n\n"
    finally:
        job.cancel()


@router.get("/admin/health")
async def health(engine: EngineDep) -> dict[str, object]:
    """The probe every client asks before anything else, and the one route an api key does
    not cover. `pid` and `uptime` are here because whoever asks may not be whoever started
    the daemon: the app draws the process it did not spawn, and the CLI decides between
    talking to a daemon and starting one."""
    return {
        "status": "ok",
        "models": engine.resident,
        "pid": os.getpid(),
        "uptime": time.monotonic() - _STARTED,
        "version": version("sideros"),
    }


@router.get("/api/openai/v1/models")
def models(store: StoreDep) -> dict[str, object]:
    """The catalog, not the residents. No dialect's schema has a notion of loaded, and a
    client cannot ask for a model it cannot see — a list of residents is empty exactly at
    boot, which is when it is read. Which of them are in memory is `/admin`'s question.

    A model with profiles answers to more than one name, and every one of them is listed:
    the dialect has no field for a profile, so an id a client cannot see is a profile it
    cannot select.

    Sync like the catalog's own handlers: the scan stats a few hundred files, and the loop
    it would otherwise sit on is carrying a token stream.
    """
    return {
        "object": "list",
        "data": [
            {"id": served, "object": "model", "created": 0, "owned_by": "sideros"}
            for served in profiles.served_ids(store)
        ],
    }


@router.post("/api/openai/v1/chat/completions", response_model=None)
async def chat(
    request: ChatRequest, engine: EngineDep, store: StoreDep
) -> JSONResponse | StreamingResponse:
    if not any(message.content for message in request.messages):
        return openai_error(400, "messages must contain non-empty content", "empty_prompt")
    if (refused := _refused(request)) is not None:
        return refused

    model_id, profile = profiles.resolve(store, request.model)
    preset = profiles.preset(model_id, profile)
    asked = _preset(request, preset)
    messages = request.messages if profile is None else _messages(request, profile.system_prompt)
    tools = _tools(request)
    checked = _checked(request)
    strict = _strict(request)
    # The conversation goes to the model as a conversation: what turns it into a prompt is
    # the checkpoint's own chat template, and a model that ships none says so below.
    try:
        turns = tuple(_turn(message) for message in messages)
    except UnreadableImage as unreadable:
        return openai_error(400, str(unreadable), "invalid_image")
    if checked is not None:
        turns = (*turns, instruction(checked.schema))
    effort = effort_of(request.reasoning_effort, preset.reasoning_effort)
    conversation = Chat(turns, tools=tools, reasoning_effort=effort)
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    spent: dict[str, int] = _counts(Meter())
    attempt = 0
    while True:
        attempt += 1
        try:
            # One walk per generation and never one shared between two: the grammar behind it
            # is the engine's to keep, the walk is this attempt's.
            constrained = None if strict is None else await engine.constrain(model_id, strict)
            # A name that does not resolve to a checkpoint and one whose load fails are the
            # same answer to the client: this model is not available here. The reason travels
            # in the message rather than in a second status code. A `model:typo` lands here
            # too, whole: no profile matched, so the name was never split.
            job = await engine.submit(
                model_id,
                conversation,
                replace(
                    _options(asked, preset, constrained),
                    speculate=profiles.speculating(store, model_id, profile),
                ),
            )
        except GrammarRefused as refusal:
            # The compiler's own words — `Unimplemented keys: ["uniqueItems"]` is a reason
            # where "grammar error" is not, and what the client does with it is send the same
            # schema without `strict` and have the answer checked instead of guaranteed.
            return openai_error(400, str(refusal), "grammar_refused")
        except NotConstrainable as refusal:
            return openai_error(400, str(refusal), "not_constrainable")
        except NotQuantizable as refusal:
            return openai_error(400, str(refusal), "not_quantizable")
        except UnsupportedInput:
            return openai_error(
                400, unsupported_reason(request.model, conversation), "unsupported_input"
            )
        except Exception as error:
            return openai_error(
                404, f"model {request.model!r} is not available: {error}", "model_not_found"
            )

        # No tools offered, nothing to read back; no family, nothing that could read it. Both
        # reach the client the way the generation always did, piece for piece: suppressing an
        # envelope no parser answers to costs the client the text and gives back no call, which
        # is what a checkpoint whose spelling this server does not know would pay.
        parser = parser_of(job.model) if tools else None
        family = None if parser is None else parser.tools
        calls = None if family is None else Calls(family)

        if request.stream:
            # The one pass a stream gets: whatever is checked is checked at the end of it,
            # inside `_events`, and there is no second attempt to come back here for.
            streaming = request.stream_options
            usage = streaming is not None and streaming.include_usage
            return StreamingResponse(
                _events(
                    job, request_id, created, request.model, usage, calls, asked.max_tokens, checked
                ),
                media_type="text/event-stream",
            )

        try:
            pieces = await _drain(job)
        finally:
            job.cancel()
        if job.error is not None:
            return openai_error(500, job.error, "generation_failed", kind="server_error")
        # Every attempt is a whole generation and the client paid for all of them: usage that
        # counted only the one that validated would disagree with `schema_attempts` about what
        # this answer cost.
        spent = {name: spent[name] + count for name, count in _counts(job.meter).items()}
        # The two channels part here for the same reason they part in the stream, and the
        # answer is what the rest of this reads: the schema is checked against what the model
        # answered, and a correction turn quotes it back — neither is about the thinking.
        thought = "".join(unmarked(p.text) for p in pieces if p.channel == "reasoning")
        answered = [piece for piece in pieces if piece.channel != "reasoning"]
        content = "".join(piece.text for piece in answered)
        made: tuple[ToolCall, ...] = ()
        if calls is not None:
            held = "".join(calls.push(piece) for piece in answered)
            tail, made = calls.finish()
            content = held + tail
        if checked is None or made:
            # A turn that called something answered with a call and not with a document: the
            # format is about the answer, and this turn has none to check.
            break
        try:
            value = document(content, checked)
        except (MalformedJSON, SchemaViolation) as violation:
            if attempt >= checked.attempts:
                reason, code = failed(violation, attempt)
                # 422 and not a 5xx: the SDKs retry a 5xx on their own, and a whole generation
                # bought behind the client's back is the cost this level exists to make visible.
                return openai_error(422, reason, code, kind="server_error")
            conversation = Chat((*turns, *_correction(content, violation)), tools=tools)
            continue
        # The document and not the text around it: the client asked for a JSON value, and prose
        # or a fence with one inside it is an answer `json.loads` refuses. What goes back is
        # what was validated.
        content = json.dumps(value, ensure_ascii=False)
        break

    message: dict[str, object] = {"role": "assistant", "content": content}
    if thought:
        message["reasoning_content"] = thought
    if made:
        # The dialect's null: a turn that only called something has no text, and `""` reads
        # as an assistant that answered with nothing.
        message["content"] = content or None
        message["tool_calls"] = [_tool_call(index, call) for index, call in enumerate(made)]
    body: dict[str, object] = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish(job, asked.max_tokens, bool(made)),
            }
        ],
        "usage": _usage(spent),
    }
    if checked is not None:
        # What the answer cost in generations, always — the default included. A retry nobody
        # sees is a model that cannot obey the schema, hidden.
        body["schema_attempts"] = attempt
    return JSONResponse(content=body)


async def _invalid_request(request: Request, error: Exception) -> JSONResponse:
    """A body no dialect's schema accepts, in the dialect of whoever sent it.

    FastAPI's own 422 with `{"detail": [...]}` is nobody's, and the SDKs do two different
    things with an error body: OpenAI's and Anthropic's pick the exception class by status and
    hand the body over raw, for the client's own error mapping to read `error.type` out of;
    Gemini's *parses* it, and prints literally `None None.` when the three keys it reads
    (`error.status`, `error.message`, `error.code`) are somebody else's.

    The route prefix chooses the encoder, the same way `auth.ApiKey` chooses one for a refusal:
    `/admin` and anything unrouted get OpenAI's, which is what every path here answered with
    before. The field is named in the message, because "invalid request" without it sends the
    client guessing which of its own fields we mean.
    """
    assert isinstance(error, RequestValidationError)
    # The deepest error and not the first: a field written inside a union — a content part, a
    # block — fails once per member of it, and the first is whichever member pydantic tried
    # first, which for an image with the wrong media type is `content.str: Input should be a
    # valid string`. The deepest is the one about the member the client meant.
    deepest = max(error.errors(), key=lambda entry: len(entry["loc"]))
    # The `loc` entries pydantic writes for a union or a list carry a `[` and our class names
    # inside them; the client's own path through its body is what is left.
    location = [str(part) for part in deepest["loc"][1:] if "[" not in str(part)]
    message = f"{'.'.join(location) or 'body'}: {deepest['msg']}"
    path = request.url.path
    if path.startswith("/api/anthropic/"):
        return anthropic.encode_error(400, "invalid_request_error", message)
    if path.startswith("/api/gemini/"):
        return gemini.error("INVALID_ARGUMENT", message)
    return openai_error(400, message, "invalid_request")


def create_app(engine: Engine, store: Store, *, host: str = "127.0.0.1") -> FastAPI:
    """The store is a parameter rather than a default: a test that builds an app must not
    write into the daemon's own file, and the daemon is the one caller that knows it wants
    `store.default_path()`.

    `host` is what the socket will be bound to, and it is here because one route needs it:
    off the loopback the api key is the only thing between the weights and the network, so
    `/admin/config` refuses to clear it. The default is the one every test wants and the one
    the daemon comes up on."""

    registry = Jobs(store)

    async def _health() -> object:
        return await health(engine)

    async def _models() -> object:
        return catalog.models(
            {model_id: entry.active_bytes for model_id, entry in engine.residency.items()}
        )

    async def _state() -> object:
        return await state.state(engine, store)

    async def _jobs() -> object:
        """The live ones. The history is a list that only grows and only one screen reads,
        and it is a fetch there — a stream is for what moves."""
        return jobs.jobs(registry, active=True)

    async def _config() -> object:
        return config.config(store)

    async def _metrics() -> object:
        return engine.metrics.snapshot()

    hub = events.Hub(
        {
            # The order is the order the opening burst arrives in, and it is the order a
            # screen needs it: what the daemon is, then what it was told, then what it has.
            "health": _health,
            "config": _config,
            "models": _models,
            "state": _state,
            "jobs": _jobs,
            "metrics": _metrics,
        }
    )
    engine.on_change = lambda: hub.publish("state")
    engine.metrics.on_change = lambda: hub.publish("metrics")
    registry.on_change = lambda: hub.publish("jobs")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        hub.bind(asyncio.get_running_loop())
        beat = asyncio.create_task(_beating(engine.metrics))
        watcher = events.observe(hub, (catalog.HUB_CACHE, catalog.QUANTIZED_CACHE))
        # Rows whose file is gone — a cache directory somebody cleared, a disk that was
        # reformatted under a database that was not. Left in, every lookup would pay for the
        # same miss and the ceiling would be enforced against bytes nobody holds.
        prefixes.sweep(store)
        engine.start()
        yield
        engine.stop()
        beat.cancel()
        events.shutdown(watcher)

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.state.host = host
    app.state.jobs = registry
    app.state.metrics = engine.metrics
    app.state.events = hub
    # Ahead of every route, including the ones no router claims: an unrouted path under a
    # dialect prefix answers 401 rather than a 404 that enumerates what is mounted.
    app.add_middleware(auth.ApiKey)
    if os.environ.get("SIDEROS_DEV") == "1":
        # Outside the key check, so a stream that was refused is a refusal on stdout too.
        # The shell already sets this for the daemon it spawns in dev.
        app.add_middleware(events.Tap)
    app.include_router(router)
    # These four before the catalog's: its `{model_id:path}` matches slashes, so it would
    # answer for `/admin/models/{id}/profiles/{name}` as if the profile were part of the
    # model's name. And `profiles` before `residency`, in the other direction: a profile
    # actually named `residency` would otherwise be read as a model id ending in `/profiles`.
    app.include_router(profiles.router)
    app.include_router(features.router)
    app.include_router(residency.router)
    app.include_router(tokenization.router)
    app.include_router(catalog.router)
    app.include_router(events.router)
    app.include_router(system.router)
    app.include_router(jobs.router)
    app.include_router(state.router)
    app.include_router(config.router)
    app.include_router(sessions.router)
    app.include_router(metrics.router)
    app.include_router(download.router)
    app.include_router(quantize.router)
    app.include_router(bench.router)
    app.include_router(benchmarks.router)
    app.include_router(datasets.router)
    app.include_router(fidelity.router)
    app.include_router(responses.router)
    app.include_router(anthropic.router)
    app.include_router(gemini.router)
    app.add_exception_handler(RequestValidationError, _invalid_request)
    return app
