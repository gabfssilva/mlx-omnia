"""The Gemini dialect: `/api/gemini/v1beta/models/{model}:{method}`.

The odd one out twice over. The method rides in the **path**, glued to the model name by a
colon, so the whole tail is the route and the split happens here rather than in the router:
an id of this house carries a `/` (`mlx-community/Qwen3-0.6B-4bit`) and may carry a `:` (the
profile suffix), and only the *last* colon is the method's. A tail that names no method we
serve is answered in this dialect's own words — a bare 404 from the router would reach the
SDK as a body it cannot parse, which is the next paragraph.

The vocabulary is the other half: `contents` of `parts`, whose role says `model` where the
rest of the world says `assistant`; the system prompt as a field of its own; the sampling
knobs under `generationConfig`, camelCased.

The error envelope is why this file writes its own instead of borrowing OpenAI's. The Google
SDK does not attach the raw body and leave the reading to the caller — it *parses* it:
`APIError` takes `error.status`, `error.message` and `error.code` out of it, and an envelope
carrying none of the three prints as `None None.`, which tells a client nothing at all about
what it got wrong.

Tools are the vocabulary again: a call is a `functionCall` part of the model's own content
and a result a `functionResponse` part of the next one, where the rest of the world has a
message per call and a message per result. And no id ties the two together — the *name*
does, which is what the turn a template renders is keyed by here.
"""

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from mlx_omnia import (
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
from mlx_omnia.chat import Effort, parser_of
from mlx_omnia.generate import Constraint, Meter
from mlx_omnia.grammar import GrammarRefused
from mlx_omnia.parsers import Segment, ToolCall
from mlx_omnia.schema import MalformedJSON, SchemaViolation
from mlx_omnia_server import profiles
from mlx_omnia_server.engine import Engine, Job, NotConstrainable, NotQuantizable
from mlx_omnia_server.profiles import Sampling, StoreDep
from mlx_omnia_server.responses import (
    Calls,
    Checked,
    ToolTurn,
    UnreadableImage,
    content_of,
    declared,
    document,
    failed,
    image_part,
    instruction,
    unsupported_reason,
)

_METHODS = ("generateContent", "streamGenerateContent")

type Status = Literal["INVALID_ARGUMENT", "NOT_FOUND", "INTERNAL", "UNAUTHENTICATED"]

_CODES: dict[Status, int] = {
    "INVALID_ARGUMENT": 400,
    "NOT_FOUND": 404,
    "INTERNAL": 500,
    "UNAUTHENTICATED": 401,
}


def envelope(status: Status, message: str) -> dict[str, object]:
    """The body of this dialect's error, which its SDK reads instead of the status line. It is
    built apart from the response because a stream carries it too: the SDK raises on a chunk
    whose JSON opens with `error`, and that is the only way to fail a generation that has
    already answered 200."""
    code = _CODES[status]
    return {"error": {"code": code, "message": message, "status": status}}


def error(status: Status, message: str) -> JSONResponse:
    """The status name is the code's canonical spelling, so the two cannot disagree — the
    caller names the one it means and the other follows."""
    return JSONResponse(status_code=_CODES[status], content=envelope(status, message))


class FunctionCall(BaseModel):
    """The call the model made, replayed by the client on the next turn. `id` is populated
    only by the models that hand one out, and this dialect never does — the name is the
    correlation key everywhere else."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, object] = Field(default_factory=dict)
    id: str | None = None


class FunctionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    response: dict[str, object]
    id: str | None = None


class InlineData(BaseModel):
    """The bytes a part carries, which for this server means an image. `mimeType` is narrowed
    to the one format read here so that anything else is refused with the name of the field
    that was wrong; the SDK base64s with the url-safe alphabet, which is where
    `responses.image_part` starts.

    `fileData` is the other half of the API's vocabulary and has no field here: it names an
    upload this server never took, and there is no file service behind it."""

    model_config = ConfigDict(extra="forbid")

    mimeType: Literal["image/png"]
    data: str


class Part(BaseModel):
    """Text, an image, a call, or the result of one — exactly one of the four, which is what
    the API documents."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    inlineData: InlineData | None = None
    functionCall: FunctionCall | None = None
    functionResponse: FunctionResponse | None = None


class Content(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[Part]
    role: Literal["user", "model"] | None = None
    """`model` is this dialect's spelling of `assistant`. Absent is `user`, which is what the
    API documents."""


class FunctionDeclaration(BaseModel):
    """One function offered to the model. The schema has two spellings and they are mutually
    exclusive upstream: `parameters` is the OpenAPI subset the SDK builds out of its `Schema`
    (types spelled `STRING`, `OBJECT`), `parametersJsonSchema` is a JSON schema as written.
    Both are the same field to a template, so both arrive at it as `parameters`.

    The second one answers to two names because the wire carries two: proto's JSON mapping
    accepts the field's own name as well as its camelCase form, the REST reference documents
    the camelCase one, and the SDK sends `parameters_json_schema`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None
    parametersJsonSchema: dict[str, object] | None = Field(
        default=None,
        validation_alias=AliasChoices("parametersJsonSchema", "parameters_json_schema"),
    )


class Tool(BaseModel):
    """Only functions. `googleSearch`, `codeExecution` and the rest of the built-ins are
    refused by name: they are executed on the vendor's side, and there is no vendor here."""

    model_config = ConfigDict(extra="forbid")

    functionDeclarations: list[FunctionDeclaration]


class FunctionCallingConfig(BaseModel):
    """`ANY` and `VALIDATED` are refused by name: both constrain decoding to a call, and
    answering `AUTO` to a client that asked for one is a call the model may never have
    made."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["AUTO", "NONE"] = "AUTO"


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functionCallingConfig: FunctionCallingConfig


class ThinkingConfig(BaseModel):
    """How long the checkpoint may think, in the one field this dialect has for it.

    `thinkingBudget` carries the switch and the length in a single number, and the three
    ranges are upstream's own: `-1` leaves the decision to the model, which is the template's
    default here and so reaches it as no kwarg at all; `0` turns thinking off; anything above
    it is thinking on with that many ids to spend, ended by feeding the block's closer once
    they are gone. There is no rung beside it because upstream has none on this route — a
    client that wants one names a profile, which is where the effort lives whole.

    `includeThoughts` is declared so that refusing it is a named error. It asks for the
    reasoning to come back as parts marked `thought`, and this route returns what the model
    wrote on one channel: honouring the `false` it defaults to would mean dropping text the
    client has been receiving, and honouring `true` would mean a part shape nothing here
    writes. Either answer would be a client told something untrue about what it got."""

    model_config = ConfigDict(extra="forbid")

    thinkingBudget: int | None = Field(default=None, ge=-1)
    includeThoughts: None = None


class GenerationConfig(BaseModel):
    """The knobs the sampler has, plus the three fields this dialect spells structured output
    in. Everything else — `stopSequences`, `candidateCount` — is refused by name: accepting
    `stopSequences` and never cutting on one answers the client with a generation it did not
    ask for."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0)
    thinkingConfig: ThinkingConfig | None = None
    topP: float | None = Field(default=None, gt=0.0, le=1.0)
    topK: int | None = Field(default=None, ge=1)
    maxOutputTokens: int = Field(default=128, gt=0)
    seed: int | None = None
    responseMimeType: Literal["text/plain", "application/json"] | None = None
    """`text/plain` is the default and asks for nothing; `application/json` asks for a JSON
    value and is checked after the generation. The other mime types the API documents —
    `text/x.enum` among them — are refused by the field's own name: what this server can
    promise about an answer is what `mlx_omnia.schema` can check and what the grammar can
    compile, and neither of them speaks those."""
    responseSchema: dict[str, object] | None = None
    """Declared so it can be refused by name rather than dropped. It is not a JSON Schema: the
    SDK builds it out of its own `Schema`, which spells the types `OBJECT` and `STRING` and
    the keys `property_ordering` and `any_of`. The schema here is compiled into a grammar and
    put into the prompt as written, so translating that spelling would be a second reading of
    a vocabulary this server does not own — `responseJsonSchema` is the field that carries the
    schema the other two dialects send."""
    responseJsonSchema: dict[str, object] | None = None
    """A JSON Schema as written, and a guarantee: the schema is compiled into a grammar and
    the ids that would break it are at -inf before the draw. This dialect has no `strict` to
    turn that off — upstream a `responseSchema` is constrained decoding too — so a schema the
    compiler will not take is refused in its own words rather than quietly checked afterwards.
    Camel-cased only, which is what the SDK writes on the wire for this one (it writes
    `parameters_json_schema` for the tool-side field, hence the two names there)."""


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contents: list[Content]
    systemInstruction: Content | None = None
    generationConfig: GenerationConfig | None = None
    tools: list[Tool] | None = None
    toolConfig: ToolConfig | None = None


def _text(content: Content) -> str:
    return "".join(part.text for part in content.parts if part.text is not None)


def _output(response: FunctionResponse) -> str:
    """What the function returned, as the characters a turn is made of. `output` and `error`
    are the two keys the API gives a meaning to; anything else is the result itself, which is
    what the documentation says in as many words."""
    payload = response.response
    value = payload.get("output", payload.get("error", payload))
    return value if isinstance(value, str) else json.dumps(value)


def _called(call: FunctionCall) -> dict[str, object]:
    """The call in the shape the templates read: `function.name` and arguments as JSON text.
    The id is the name when the client sent none, which is the usual case — it is what a
    `functionResponse` is matched to its call by here."""
    return {
        "id": call.id or call.name,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.args)},
    }


def _parts(content: Content) -> list[TextPart | ImagePart]:
    """The text and the images of one content, in the order they arrived in: where an image
    sits among the words is what the template writes a marker for, and what the model then
    looks at."""
    parts: list[TextPart | ImagePart] = []
    for part in content.parts:
        if part.text is not None:
            parts.append({"type": "text", "text": part.text})
        if part.inlineData is not None:
            parts.append(image_part(part.inlineData.data, part.inlineData.mimeType))
    return parts


def _turns(content: Content) -> list[ToolTurn]:
    """One content, and the turns it spells. Its results come first because they are the
    round the content answers, and the text after them is the client's next word."""
    said = content_of(_parts(content))
    made = [_called(part.functionCall) for part in content.parts if part.functionCall is not None]
    answered = [
        part.functionResponse for part in content.parts if part.functionResponse is not None
    ]
    turns: list[ToolTurn] = [
        {"role": "tool", "content": _output(response), "tool_call_id": response.id or response.name}
        for response in answered
    ]
    if said or made or not answered:
        turn: ToolTurn = {
            "role": "assistant" if content.role == "model" else "user",
            "content": said,
        }
        if made:
            turn["tool_calls"] = made
        turns.append(turn)
    return turns


def _messages(body: GenerateRequest, system_prompt: str | None) -> tuple[ToolTurn, ...]:
    """The instruction the request carries wins over the profile's — the same precedence the
    knobs below follow — and there is at most one either way, so no template is left with two
    system turns to pick between."""
    instruction = body.systemInstruction
    system = _text(instruction) if instruction is not None else system_prompt
    turns = [turn for content in body.contents for turn in _turns(content)]
    return tuple(turns) if system is None else ({"role": "system", "content": system}, *turns)


def _tools(body: GenerateRequest) -> tuple[Mapping[str, object], ...]:
    """`mode: NONE` is honoured where it can be honoured: the declarations never enter the
    prompt, so the model has nothing to call rather than an instruction not to."""
    config = body.toolConfig
    if body.tools is None or (config is not None and config.functionCallingConfig.mode == "NONE"):
        return ()
    return tuple(
        declared(
            declaration.name,
            declaration.description,
            declaration.parameters
            if declaration.parameters is not None
            else declaration.parametersJsonSchema,
        )
        for tool in body.tools
        for declaration in tool.functionDeclarations
    )


def _knob(asked: float | None, preset: float | None, default: float) -> float:
    """The request's value, then the profile's, then the dialect's default: a knob the client
    named it meant, whatever profile it also named. `min_p` and the repetition penalty have
    no field in this dialect, so for those two the request never names anything and the
    profile is the only thing that can set them."""
    if asked is not None:
        return asked
    return default if preset is None else preset


def _thinks(asked: GenerationConfig, preset: Sampling) -> Effort:
    """What the template is told about thinking, off the one number this dialect has for it.

    `-1` and an absent `thinkingConfig` are the same thing — the decision is the model's — so
    both fall to the profile and then to `auto`. `0` is off. Anything above it is on, and the
    number itself is the budget rather than a rung: this dialect names a length, and turning
    one into `medium` would write a level into the prompt that no client asked for."""
    config = asked.thinkingConfig
    budget = None if config is None else config.thinkingBudget
    if budget is None or budget < 0:
        return "auto" if preset.reasoning_effort is None else preset.reasoning_effort
    return "off" if budget == 0 else "on"


def _budget(asked: GenerationConfig, preset: Sampling) -> int | None:
    """The ids the block may spend. `-1` is no cap and `0` is a block that never opens, so
    neither is a number the loop counts down — what reaches it is a positive budget, or the
    profile's when the request named none."""
    config = asked.thinkingConfig
    budget = None if config is None else config.thinkingBudget
    if budget is None:
        return preset.reasoning_budget
    return budget if budget > 0 else None


def _options(
    asked: GenerationConfig, preset: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """Filters in the order the cuts expect: they read the distribution temperature already
    shaped, which is what makes `topP` here mean what it means upstream.

    The constraint composes with all of them and is nobody's filter: the mask is applied
    before the sampler runs, so what is drawn is drawn from what the grammar left."""
    repeats = _knob(None, preset.repetition_penalty, 1.0)
    penalty: Penalty | None = None if repeats == 1.0 else repetition_penalty(repeats)
    heat = _knob(asked.temperature, preset.temperature, 1.0)
    budget = _budget(asked, preset)
    if heat == 0.0:
        # The deterministic end of the dial: nothing is left to draw from, and dividing by it
        # would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=asked.maxOutputTokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=budget,
        )

    filters: list[LogitFilter] = [temperature(heat)]
    cut = asked.topK if asked.topK is not None else preset.top_k
    if cut is not None:
        filters.append(top_k(cut))
    nucleus = _knob(asked.topP, preset.top_p, 1.0)
    if nucleus < 1.0:
        filters.append(top_p(nucleus))
    floor = _knob(None, preset.min_p, 0.0)
    if floor > 0.0:
        filters.append(min_p(floor))
    seed = asked.seed if asked.seed is not None else preset.seed
    drawn: Sampler = sampler(*filters, seed=seed)
    return GenerationOptions(
        max_tokens=asked.maxOutputTokens,
        sampler=drawn,
        penalty=penalty,
        constraint=constraint,
        reasoning_budget=budget,
    )


def _refused(asked: GenerationConfig, tools: tuple[Mapping[str, object], ...]) -> str | None:
    """The three shapes of structured output this dialect can spell and this route cannot
    honour, each named.

    The first is a schema in the SDK's own `Schema` spelling (see `responseSchema`). The
    second is the API's own rule — a schema needs the mime type beside it — and honouring it
    without one would answer a request for JSON with prose around a document. The third is
    the schema against the tools: the grammar constrains decoding to it from the first token,
    so the model cannot write a call however it is offered, and a 200 with no calls in it is
    that told as a success."""
    if asked.responseSchema is not None:
        return (
            "responseSchema is the OpenAPI subset the SDK builds out of its own Schema — "
            "types spelled OBJECT and STRING — and what is compiled into a grammar here is a "
            "JSON Schema. Send it as responseJsonSchema."
        )
    if asked.responseJsonSchema is None:
        return None
    if asked.responseMimeType != "application/json":
        return (
            "responseJsonSchema needs responseMimeType 'application/json' beside it: a schema "
            "without it asks for a document and for prose around it at the same time."
        )
    if tools:
        return (
            "responseJsonSchema and tools cannot both be honoured: the grammar constrains "
            "decoding to the schema from the first token, so the model cannot write a call "
            "however it is offered. Drop the schema, or offer no functions."
        )
    return None


def _call(call: ToolCall) -> dict[str, object]:
    """A call, as a part of the model's own content — this dialect's `tool_calls`, and the
    reason a candidate here can carry parts of two kinds at once."""
    return {"functionCall": {"name": call.name, "args": call.arguments}}


def _reply(
    model: str, parts: list[dict[str, object]], meter: Meter | None, budget: int = 0
) -> dict[str, object]:
    """One `GenerateContentResponse`: the whole answer for `generateContent`, one piece of it
    for a stream frame. A meter means this is the last of them — the finish reason and the
    counts are the request's, and a frame that published them early would be publishing a
    partial total as a total.

    `MAX_TOKENS` where the budget ran out, which is the branch an agent loop takes to
    continue: `STOP` over a sentence `maxOutputTokens` cut is a truncation reported as an
    answer."""
    candidate: dict[str, object] = {
        "content": {"role": "model", "parts": parts},
        "index": 0,
    }
    payload: dict[str, object] = {"candidates": [candidate], "modelVersion": model}
    if meter is not None:
        cut = budget > 0 and meter.completion_tokens >= budget
        candidate["finishReason"] = "MAX_TOKENS" if cut else "STOP"
        payload["usageMetadata"] = {
            "promptTokenCount": meter.prompt_tokens,
            # A subset of the prompt, the way this dialect counts and the way the OpenAI one
            # does — not a fourth addend as in the Anthropic dialect. Written even at zero:
            # the field absent says the server does not carry it, and this one does.
            "cachedContentTokenCount": meter.reused_tokens,
            "candidatesTokenCount": meter.completion_tokens,
            "totalTokenCount": meter.prompt_tokens + meter.completion_tokens,
        }
    return payload


async def _drain(job: Job) -> list[Segment]:
    pieces: list[Segment] = []
    while (piece := await job.chunks.get()) is not None:
        pieces.append(piece)
    return pieces


async def _events(
    job: Job, model: str, calls: Calls | None, budget: int, checked: Checked | None
) -> AsyncIterator[str]:
    """No keep-alive rides this stream. The SDK's reader takes every line that is not a
    `data:` one as part of a JSON body to accumulate, so an SSE comment reaches `json.loads`
    and raises at the client instead of being skipped — a long prefill is silent here by
    necessity.

    The calls ride the closing frame, whole and all of them: the one frame that already
    carries the finish reason is the one a client reads a completed turn off. A11's decision.
    """
    sent: list[str] = []
    try:
        while (piece := await job.chunks.get()) is not None:
            # No family, nothing that could read a channel back: every segment is text the
            # client asked for, whichever one the model wrote it on.
            if content := (piece.text if calls is None else calls.push(piece)):
                sent.append(content)
                yield f"data: {json.dumps(_reply(model, [{'text': content}], None))}\n\n"
        parts: list[dict[str, object]] = []
        made: tuple[ToolCall, ...] = ()
        if calls is not None:
            tail, made = calls.finish()
            if tail:
                sent.append(tail)
                parts.append({"text": tail})
            parts += [_call(call) for call in made]
        if job.error is not None:
            # This dialect's error, on the wire, as the last frame: the SDK reads a chunk that
            # opens with `error` and raises on it. Without this the closing frame below would
            # carry `finishReason: "STOP"` and a `usageMetadata` counted off a generation that
            # died — a failure wearing the shape of an answer, which is the one thing a client
            # cannot tell apart afterwards.
            yield f"data: {json.dumps(envelope('INTERNAL', job.error))}\n\n"
            return
        if checked is not None and not made:
            try:
                document("".join(sent), checked)
            except (MalformedJSON, SchemaViolation) as violation:
                # A stream cannot take back the text it already handed out, and this dialect
                # sells no second attempt: what is left is the same door a generation that
                # died goes through, which is the only way to fail a request that already
                # answered 200. The frames stay as they were sent — what failed is the
                # answer, not the client's reading of it.
                reason, _ = failed(violation, checked.attempts)
                yield f"data: {json.dumps(envelope('INTERNAL', reason))}\n\n"
                return
        yield f"data: {json.dumps(_reply(model, parts or [{'text': ''}], job.meter, budget))}\n\n"
    finally:
        job.cancel()


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


@router.get("/api/gemini/v1beta/models")
def models(store: StoreDep) -> dict[str, object]:
    """The catalog under this dialect's name for a model — `models/{id}`, which is what the
    SDK puts back into the path when a client asks for one. Profiles are listed too: no
    dialect has a field for a profile, so an id a client cannot see is a preset it cannot
    select."""
    return {
        "models": [
            {"name": f"models/{served}", "supportedGenerationMethods": list(_METHODS)}
            for served in profiles.served_ids(store)
        ]
    }


@router.post("/api/gemini/v1beta/models/{tail:path}", response_model=None)
async def generate(
    tail: str, body: GenerateRequest, engine: EngineDep, store: StoreDep
) -> JSONResponse | StreamingResponse:
    """The tail is the model name and the method glued by a colon, and the split is at the
    *last* one: `qwen:code:generateContent` selects the profile `code`, and a checkpoint path
    with a colon of its own keeps it."""
    name, _, method = tail.rpartition(":")
    if method not in _METHODS:
        return error(
            "NOT_FOUND",
            f"{tail!r} names no method this dialect serves: a path here ends in "
            "':generateContent' or ':streamGenerateContent'",
        )

    model_id, profile = profiles.resolve(store, name)
    tools = _tools(body)
    asked = body.generationConfig if body.generationConfig is not None else GenerationConfig()
    if (refusal := _refused(asked, tools)) is not None:
        return error("INVALID_ARGUMENT", refusal)
    # The two levels this dialect can ask for, and never both: a schema is a guarantee here —
    # upstream it is constrained decoding too — while the mime type on its own is a demand
    # for JSON that is measured once the generation is spent.
    strict = asked.responseJsonSchema
    wants_json = asked.responseMimeType == "application/json"
    checked = Checked(None, 1) if strict is None and wants_json else None
    try:
        turns = _messages(body, None if profile is None else profile.system_prompt)
    except UnreadableImage as unreadable:
        return error("INVALID_ARGUMENT", str(unreadable))
    if checked is not None:
        turns = (*turns, instruction(checked.schema))
    preset = profiles.preset(model_id, profile)
    conversation = Chat(turns, tools=tools, reasoning_effort=_thinks(asked, preset))
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        constrained = None if strict is None else await engine.constrain(model_id, strict)
        # A name no checkpoint answers to and one whose load fails are the same answer to the
        # client: this model is not available here, with the reason in the message rather than
        # in a second status. A `model:typo` arrives here whole — no profile matched, so the
        # name was never split.
        job = await engine.submit(
                model_id,
                conversation,
                replace(
                    _options(asked, preset, constrained),
                    speculate=profiles.speculating(store, model_id, profile),
                ),
            )
    except GrammarRefused as refusal:
        # The compiler's own words — `Unimplemented keys: ["uniqueItems"]` is a reason where
        # "grammar error" is not, and it is what tells the client which keyword to drop.
        return error("INVALID_ARGUMENT", str(refusal))
    except NotConstrainable as refusal:
        return error("INVALID_ARGUMENT", str(refusal))
    except NotQuantizable as refusal:
        return error("INVALID_ARGUMENT", str(refusal))
    except UnsupportedInput:
        return error("INVALID_ARGUMENT", unsupported_reason(name, conversation))
    except Exception as failure:
        return error("NOT_FOUND", f"model {name!r} is not available: {failure}")

    # No tools offered, nothing to read back; no family, nothing that could read it. Both
    # reach the client the way the generation always did, piece for piece: suppressing an
    # envelope no parser answers to costs the client the text and gives back no call.
    parser = parser_of(job.model) if tools else None
    family = None if parser is None else parser.tools
    calls = None if family is None else Calls(family)

    if method == "streamGenerateContent":
        return StreamingResponse(
            _events(job, name, calls, asked.maxOutputTokens, checked),
            media_type="text/event-stream",
        )

    try:
        pieces = await _drain(job)
    finally:
        job.cancel()
    if job.error is not None:
        return error("INTERNAL", job.error)
    text = "".join(piece.text for piece in pieces)
    made: tuple[ToolCall, ...] = ()
    if calls is not None:
        held = "".join(calls.push(piece) for piece in pieces)
        tail, made = calls.finish()
        text = held + tail
    # A content that called something answered with a call and not with a document: the mime
    # type is about the answer, and this turn has none to check.
    if checked is not None and not made:
        try:
            value = document(text, checked)
        except (MalformedJSON, SchemaViolation) as violation:
            # INTERNAL and not INVALID_ARGUMENT: what did not validate is what this server
            # generated, and this dialect has four statuses of which only one does not blame
            # the client for it. The SDK retries nothing by default, so the generation is not
            # bought again behind the client's back.
            reason, _ = failed(violation, checked.attempts)
            return error("INTERNAL", reason)
        # The document and not the text around it: the client asked for JSON, and prose or a
        # fence with a value inside it is an answer `json.loads` refuses.
        text = json.dumps(value, ensure_ascii=False)
    # A content that only called something carries no text part: an empty one is a model
    # that answered with nothing, which is not what happened.
    parts: list[dict[str, object]] = [] if made and not text else [{"text": text}]
    parts += [_call(call) for call in made]
    return JSONResponse(content=_reply(name, parts, job.meter, asked.maxOutputTokens))
