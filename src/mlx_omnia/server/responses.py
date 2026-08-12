"""`POST /api/openai/v1/responses`: the OpenAI dialect's other generation route.

The same generation as `chat/completions` behind two differences that go all the way down.
The input is `input` — a string, or a list of typed items — with `instructions` as a field
of its own rather than a message; the conversation carries it as its first system turn,
because the checkpoint's template has exactly one place to put it. And the stream is a
sequence of *named* events over one output item instead of `choices[].delta`: the SDK's
accumulator rebuilds the `Response` out of them and refuses anything before
`response.created`, so the opening and closing frames are the contract, not decoration.

What this route does not do is the half of the API that is state. `store` — and the
conversation ids that hang off it — would have the server answering for text it never kept,
so a body that asks for it is refused by name. Everything else the dialect has and this
route has not, `previous_response_id` among them, is refused by `extra="forbid"` for the
same reason: an ignored field is a client told, wrongly, that it was honoured.

`Calls` lives here rather than beside one route: reading the tool channel of a generation off
the segments the stream labelled is the same job in every dialect, and what differs — the
envelope it goes out in — is each route's own. The tools go the other way through the same
door: whatever shape a dialect spells
a function in, what reaches the template is OpenAI's nested one, which is what transformers
documents and what makes one conversation render the same through all four. So does an image:
`inline_image` and `image_part` are the base64 each dialect carries one in, and `content_of`
the turn all four build out of it.

Structured output is here for the same reason, and it is `Checked`, `instruction`, `document`
and `failed`. All four dialects spell the field differently — `response_format`, `text.format`,
`generationConfig.responseMimeType`, `output_config.format` — and none of them differs in what
the answer is measured against: level 1 is the schema as the last turn of the prompt and
`mlx_omnia.engine.schema` over the text that came back. What stays in each route is the shape of the
field and the envelope a refusal travels in. The retry does not: only `chat/completions` sells
a second generation, so the correction turns live there with the field that counts them.
"""

import asyncio
import base64
import binascii
import json
import time
import uuid
import zlib
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import count
from typing import Annotated, Final, Literal

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia import (
    Chat,
    GenerationOptions,
    Image,
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
from mlx_omnia import ChatMessage as Turn
from mlx_omnia.engine.chat import Effort, parser_of
from mlx_omnia.engine.generate import Constraint, Meter
from mlx_omnia.engine.grammar import GrammarRefused
from mlx_omnia.engine.parsers import MalformedToolCall, Segment, ToolCall, ToolFamily
from mlx_omnia.engine.schema import (
    MalformedJSON,
    SchemaViolation,
    extract_json,
    json_instruction,
    validate,
)
from mlx_omnia.server import profiles
from mlx_omnia.server.engine import Engine, Job, NotConstrainable, NotQuantizable
from mlx_omnia.server.profiles import Sampling, StoreDep

_KEEP_ALIVE_SECONDS = 0.5


class ToolTurn(Turn, total=False):
    """The two keys a conversation that used tools carries and a plain one does not. The
    template iterates the messages and reads what is in each dict: the assistant's own calls
    when it replays the turn that made them, and the id of the call a result answers."""

    tool_calls: list[dict[str, object]]
    tool_call_id: str


class Calls:
    """The tool channel of one generation, off the channel the stream already labelled.

    It reads `Segment.channel` instead of matching markers again. What decides the channel is
    the `Segmenter` inside the streamer, and it knows two things this side cannot: the channel
    the rendered prompt left open — Qwen's template writes `<think>` into it, so the first
    token is already inside the block — and whether the checkpoint's template spells a family
    at all. A second machine here would answer differently about the same text, and an
    envelope the model wrote while reasoning is the case where it does.

    The family is still needed, and only for `parse_tool_call`: `parser_of` reads the dialect
    off the source the capability compiled, so a checkpoint whose spelling nothing here parses
    never reaches this class.

    Reasoning is the dialect's to name, not this class's: `chat/completions` answers with
    `reasoning_content` and hands this only what is left, while the three dialects that have no
    field for it yet pass the block through as content — which is where it came out before. On
    either route what this holds back is the tool channel and nothing else.
    """

    def __init__(self, family: ToolFamily) -> None:
        self._family = family
        self._envelopes: list[str] = []

    def push(self, segment: Segment) -> str:
        """The content of `segment` that is safe to hand out now — nothing, when it is an
        envelope this holds on to until the generation ends."""
        if segment.channel == "tool":
            self._envelopes.append(segment.text)
            return ""
        return segment.text

    def finish(self) -> tuple[str, tuple[ToolCall, ...]]:
        """What the end of the generation releases: the calls the envelopes spell, or those
        envelopes as content when they spell none. An envelope that spells no call goes back
        into the content, at the end of it — it was held as a possible call and it is not one,
        and text held and then dropped reaches the client as a model that chose to call
        nothing, which is the shape of a refusal. All of them or none: a turn whose calls
        cannot be read whole is not a turn to half-execute."""
        calls = self._parsed()
        return ("", calls) if calls else ("".join(self._envelopes), ())

    def _parsed(self) -> tuple[ToolCall, ...]:
        try:
            return tuple(
                call
                for envelope in self._envelopes
                for call in self._family.parse_tool_call(envelope)
            )
        except MalformedToolCall:
            return ()


class UnreadableImage(ValueError):
    """An attachment that cannot become pixels. The message is the client's — it says what was
    wrong with what arrived, and each dialect puts it in its own envelope."""


_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
"""Bytes per pixel by png colour type: grey, rgb, palette index, grey+alpha, rgba."""

_STANDARD = str.maketrans("-_", "+/")
"""The url-safe base64 alphabet back to the standard one. The Gemini SDK encodes bytes with
`-_` (`ser_json_bytes='base64'` over `urlsafe_b64encode`), the other two with `+/`, and the
same image has to reach the model either way."""


def _bytes(payload: str) -> bytes:
    """`validate=True`, and therefore the translation above first: the permissive default
    *drops* a character outside the alphabet instead of failing, and one `_` silently gone
    shifts every byte after it — an image that decodes to noise rather than to an error."""
    try:
        return base64.b64decode("".join(payload.split()).translate(_STANDARD), validate=True)
    except binascii.Error as bad:
        raise UnreadableImage(f"the image is not base64: {bad}") from bad


def _unfiltered(raw: bytes, height: int, stride: int, step: int) -> bytes:
    """The five row filters of the format, undone. A row is predicted from the row above it and
    a byte from the pixel to its left, so this is sequential by construction — ~30 ms for a
    448x352 image, and the reason a very large png costs the event loop something."""
    out = bytearray(height * stride)
    previous = bytes(stride)
    at = 0
    for row in range(height):
        kind = raw[at]
        line = bytearray(raw[at + 1 : at + 1 + stride])
        at += 1 + stride
        if kind == 1:
            for i in range(step, stride):
                line[i] = (line[i] + line[i - step]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                up = previous[i]
                corner = previous[i - step] if i >= step else 0
                guess = left + up - corner
                by_left, by_up, by_corner = (
                    abs(guess - left),
                    abs(guess - up),
                    abs(guess - corner),
                )
                if by_left <= by_up and by_left <= by_corner:
                    near = left
                elif by_up <= by_corner:
                    near = up
                else:
                    near = corner
                line[i] = (line[i] + near) & 0xFF
        elif kind != 0:
            raise UnreadableImage(f"row {row} of the png names filter {kind} and there are five")
        out[row * stride : (row + 1) * stride] = line
        previous = bytes(line)
    return bytes(out)


def _pixels(png: bytes) -> npt.NDArray[np.uint8]:
    """An 8-bit png as the `[h, w, 3]` bytes `process_image` takes.

    Read here because nothing else in the tree reads an encoded image: the engine's processor
    starts at the pixels and all three dialects hand over a file. Png only, and no guessing —
    a jpeg is refused by name rather than read as noise. 16-bit and interlaced are refused for
    the same reason a guessed chat template is: a branch no test exercises is a wrong answer
    waiting, and neither reaches this server from a client that can send a png instead.

    Alpha is dropped rather than composited, which is what `PIL.Image.convert("RGB")` does —
    the reference every vision fixture in this repo was made against.
    """
    if not png.startswith(_SIGNATURE):
        raise UnreadableImage("only png is read here, and these bytes carry no png header")
    header: tuple[int, int, int, int, int] | None = None
    palette = b""
    body: list[bytes] = []
    at = len(_SIGNATURE)
    while at + 8 <= len(png):
        length = int.from_bytes(png[at : at + 4])
        kind = png[at + 4 : at + 8]
        payload = png[at + 8 : at + 8 + length]
        at += 12 + length
        if kind == b"IHDR" and len(payload) == 13:
            header = (
                int.from_bytes(payload[:4]),
                int.from_bytes(payload[4:8]),
                payload[8],
                payload[9],
                payload[12],
            )
        elif kind == b"PLTE":
            palette = payload
        elif kind == b"IDAT":
            body.append(payload)
        elif kind == b"IEND":
            break
    if header is None:
        raise UnreadableImage("the png carries no header")
    width, height, depth, colour, interlace = header
    if depth != 8 or interlace != 0 or colour not in _CHANNELS:
        raise UnreadableImage(
            f"this png is not read here: {depth}-bit, colour type {colour}"
            f"{', interlaced' if interlace else ''}"
        )
    if not width or not height:
        raise UnreadableImage("the png has no pixels")
    channels = _CHANNELS[colour]
    stride = width * channels
    try:
        raw = zlib.decompress(b"".join(body))
    except zlib.error as bad:
        raise UnreadableImage(f"the png's pixels do not decompress: {bad}") from bad
    if len(raw) != height * (stride + 1):
        raise UnreadableImage("the png ends before its last row")
    rows = np.frombuffer(_unfiltered(raw, height, stride, channels), dtype=np.uint8)
    grid = rows.reshape(height, width, channels)
    if colour == 3:
        table = np.frombuffer(palette[: len(palette) - len(palette) % 3], dtype=np.uint8)
        entries = table.reshape(-1, 3)
        if int(grid.max()) >= len(entries):
            raise UnreadableImage("the png's palette is shorter than the indices in it")
        return entries[grid[:, :, 0]]
    return np.repeat(grid[:, :, :1], 3, axis=2) if channels < 3 else grid[:, :, :3]


def image_part(payload: str, media_type: str) -> ImagePart:
    """One image on its way into a conversation, out of the base64 a dialect carried it in."""
    if media_type != "image/png":
        raise UnreadableImage(f"{media_type} is not read here: attach the image as image/png")
    return {"type": "image", "image": Image(_pixels(_bytes(payload)))}


def inline_image(url: str) -> ImagePart:
    """OpenAI's two routes spell an image as a URL, and the only one read here is the `data:`
    one that carries the bytes: fetching an `https://` one would have the daemon making
    requests of its own, at a client's word, from inside the network it was told to serve."""
    head, _, payload = url.partition(",")
    if not head.startswith("data:") or not head.endswith(";base64") or not payload:
        raise UnreadableImage(
            "an image travels here as a data URL with the bytes in it "
            "(`data:image/png;base64,…`): nothing fetches a remote image"
        )
    return image_part(payload, head.removeprefix("data:").removesuffix(";base64"))


def content_of(parts: Sequence[TextPart | ImagePart]) -> str | tuple[TextPart | ImagePart, ...]:
    """A turn's content as the template will read it: the characters when there is no image in
    it, the parts themselves when there is.

    Not a list either way, because a template concatenates `content` — `{{ message['content'] }}`
    over a one-element list writes the list's repr into the prompt — and only the vision branch
    of the templates in circulation iterates it (`content is iterable and content is not
    mapping`). A conversation of text renders through the same templates it always did.
    """
    texts = [part["text"] for part in parts if part["type"] == "text"]
    return "".join(texts) if len(texts) == len(parts) else tuple(parts)


def unsupported_reason(model: str, conversation: Chat) -> str:
    """Why a model took no conversation, in words a client can act on. Both refusals are the
    same `UnsupportedInput`: a checkpoint that ships no chat template takes no conversation at
    all, and a text-only one takes every conversation but this, because there is an image in
    it. The image is named first — for a checkpoint with neither template nor vision tower both
    lines are true, and the one about what the client just attached is the useful one."""
    for message in conversation.messages:
        content = message["content"]
        if not isinstance(content, str) and any(part["type"] == "image" for part in content):
            return f"model {model!r} does not accept an image: the checkpoint has no vision tower"
    return f"model {model!r} does not accept a conversation: the checkpoint ships no chat template"


@dataclass(frozen=True)
class Checked:
    """What one request asks to be checked: the schema the answer is measured against —
    `None` is `json_object`, which asks only that the answer be JSON — and how many whole
    generations the client agreed to pay for.

    One is what three of the four dialects can say: only `chat/completions` has a field for
    the count and a place in its body to report it back, and a generation bought behind the
    client's back is what this level exists to make visible."""

    schema: Mapping[str, object] | None
    attempts: int


def instruction(schema: Mapping[str, object] | None) -> Turn:
    """The schema as the model reads it: a turn like any other, and the last one. What the
    template renders is the conversation, so an instruction placed after the client's own turns
    is the nearest thing to the answer — and the text is the engine's `json_instruction`, so
    what is asked for and what `validate` enforces cannot drift into two readings."""
    return {"role": "system", "content": json_instruction(schema)}


def document(content: str, checked: Checked) -> object:
    """The JSON value the answer carries, against the schema when there is one. Raises
    `MalformedJSON` or `SchemaViolation`, which is what makes a violation a failure rather than
    an answer with something wrong in it."""
    value = extract_json(content)
    if checked.schema is not None:
        validate(value, checked.schema)
    return value


def failed(failure: MalformedJSON | SchemaViolation, attempts: int) -> tuple[str, str]:
    """What the client is told about an answer that did not validate, and under which code.
    The generations spent are in the message: what this level costs is interactions, and a
    client that cannot see how many it bought cannot decide whether to buy more."""
    spent = f"after {attempts} generation{'' if attempts == 1 else 's'}"
    if isinstance(failure, SchemaViolation):
        where = f"{failure.path} {failure.reason}"
        message = f"the answer does not validate against the schema {spent}: {where}"
        return message, "schema_violation"
    return f"the answer carries no JSON value {spent}", "malformed_json"


class ContentPart(BaseModel):
    """One part of an item's content. `output_text` is the type this route writes, so an
    item replayed out of a previous answer travels back in the shape it left in."""

    type: Literal["input_text", "output_text"]
    text: str


class InputImage(BaseModel):
    """This dialect's image part: the URL flat on the part rather than nested under
    `image_url`, which is `chat/completions`'s shape. `detail` is required by the SDK's own
    parameter type, so it is declared here to be refused by name rather than dropped — what an
    image costs is the checkpoint processor's decision, and a client told `high` was honoured
    was told nothing."""

    type: Literal["input_image"]
    image_url: str
    detail: Literal["auto"] = "auto"


class MessageItem(BaseModel):
    """Unknown fields are dropped here rather than refused, unlike the request's own: an
    item a client replays is the item the dialect wrote, which carries an `id`, a `status`
    and the part's `annotations`, none of which means anything on the way back."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[Annotated[ContentPart | InputImage, Field(discriminator="type")]]
    """The discriminator is what makes a refusal worth reading: without it a part with one
    field wrong fails once per member of the union, and what the client is told about is
    whichever member pydantic tried first."""


class FunctionCallItem(BaseModel):
    """The call this route wrote, replayed. `call_id` is the handle the pair is matched by —
    the item's own `id` is dropped with the rest, because nothing here is asked to find an
    item again."""

    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionOutputItem(BaseModel):
    """What the client's own function returned. `output` is text: what the template renders
    is a turn, and a turn is characters."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


type InputItem = MessageItem | FunctionCallItem | FunctionOutputItem


class FunctionTool(BaseModel):
    """This dialect spells a function flat, where `chat/completions` nests it under
    `function`. `strict` is declared so it can be refused rather than dropped: the SDK's own
    parameter type requires the key on every tool, so a client that never asked for strict
    validation still sends it."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None
    strict: bool | None = None


class TextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]


class JsonObjectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class SchemaOutput(BaseModel):
    """This dialect's `json_schema`, and the shape is the difference: the name, the schema and
    `strict` sit *on* the format, where `chat/completions` nests them under a `json_schema`
    object. The schema arrives under `schema`, which is an alias because a pydantic field of
    that name shadows `BaseModel.schema`.

    `strict` is what tells the two levels apart: with it the schema is compiled into a grammar
    and decoding cannot produce a violation, without it the schema enters the prompt and the
    answer is checked afterwards. A schema the compiler will not take is refused in its own
    words rather than quietly demoted — the client asked for a guarantee."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    name: str
    definition: dict[str, object] = Field(alias="schema")
    """Required here, where `chat/completions` lets it be absent: this dialect's own type has
    it on every `json_schema` format, so a body without one is refused by the field's name."""
    description: str | None = None
    strict: bool | None = None


type OutputFormat = TextOutput | JsonObjectOutput | SchemaOutput


class TextConfig(BaseModel):
    """`text` is where this dialect puts the format. `verbosity` is the other key it has and
    this route has not, and it is refused with the rest of them (see the module docstring):
    it decides how much the model writes, and a client told it was honoured was told
    nothing."""

    model_config = ConfigDict(extra="forbid")

    format: Annotated[OutputFormat, Field(discriminator="type")] | None = None
    """The discriminator is what makes a refusal worth reading: without it a format with one
    field wrong fails once per member of the union, and the client is told about whichever
    member pydantic tried first."""


type OpenAIEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""How the two OpenAI dialects spell an effort, which is upstream's vocabulary widened by
two rungs. `xhigh` and `max` are not in the OpenAI spec — they are what Anthropic and
DeepSeek V4 call the rungs above `high`, and the templates that read `reasoning_effort`
spell whatever they are handed — so accepting them here is a superset and never a lie: a
level named is a level rendered.

The one collapse is `minimal`, which upstream added for GPT-5 and no template in
circulation reads: it lands on `low`, the nearest rung that means something, rather than
being refused to a client whose SDK sends it by default."""

_EFFORT: Final[Mapping[OpenAIEffort, Effort]] = {
    "none": "off",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def effort_of(asked: OpenAIEffort | None, preset: Effort | None) -> Effort:
    """The effort this generation runs at: the request's, then the profile's, then `auto`.

    Same precedence as every other knob — a request that names one means it — and `auto` is
    what a request and a profile that both said nothing add up to, which leaves the decision
    with the checkpoint's own template.
    """
    if asked is not None:
        return _EFFORT[asked]
    return "auto" if preset is None else preset


class Reasoning(BaseModel):
    """`reasoning.effort` is this dialect's spelling of the same knob `chat/completions` puts
    at the top level.

    `summary` is declared so that refusing it is a named error: it asks for the reasoning
    back as a summary, there is no summarizer here, and answering with the raw block under
    that name would be a client told it received something shorter than it did. There is no
    budget beside it because upstream has none — how long the model may think is reachable
    on this route through a profile, which is where the knobs this dialect cannot spell
    already live."""

    model_config = ConfigDict(extra="forbid")

    effort: OpenAIEffort | None = None
    summary: None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[InputItem]
    instructions: str | None = None
    text: TextConfig | None = None
    """Which of the two levels this answer gets, or neither. Without `strict` the schema enters
    the prompt as a turn and the answer is checked against it once the generation is spent,
    which is a check that can fail; with it the schema is compiled into a grammar and the mask
    makes the violation unreachable, at the price of a sync per step.

    One generation either way: the second attempt `chat/completions` sells is a field of that
    dialect, and there is nowhere here to ask for one or to read back what it cost."""
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["none", "auto"] = "auto"
    """`required` and a named function are refused by name: forcing a call is a constraint on
    decoding, and accepting the field without one answers with a call the model may never
    have made."""
    store: bool = False
    """`false` is the only answer this server can give truthfully — see the module
    docstring. The field is declared so that saying so is a named error and not the generic
    refusal an undeclared one would get."""
    max_output_tokens: int = Field(default=128, gt=0)
    reasoning: Reasoning | None = None
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = None


def _options(
    request: ResponsesRequest, sampling: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """The dialect's defaults are OpenAI's, so an unset `temperature` is 1.0 and the answer
    is drawn, not argmaxed. Filters run in the order below — the cuts read the distribution
    temperature already shaped.

    The constraint composes with all of them and is nobody's filter: the mask is applied
    before the sampler runs, so what is drawn is drawn from what the grammar left.

    `sampling` is here for the knobs this dialect has no field for — `reasoning_budget` is
    the only one so far — which `_preset` cannot fill because there is nothing to fill."""
    repeats = request.repetition_penalty
    penalty: Penalty | None = None if repeats == 1.0 else repetition_penalty(repeats)
    thinking = sampling.reasoning_budget
    if request.temperature == 0.0:
        # The deterministic end of the dial: no distribution is left to draw from, and
        # dividing by it would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=request.max_output_tokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=thinking,
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
        max_tokens=request.max_output_tokens,
        sampler=drawn,
        penalty=penalty,
        constraint=constraint,
        reasoning_budget=thinking,
    )


def _wanted(request: ResponsesRequest) -> JsonObjectOutput | SchemaOutput | None:
    """What this request asks of the answer, or `None` when it asks for nothing — `text` is
    the dialect's own default and says exactly that."""
    text = request.text
    asked = None if text is None else text.format
    return None if isinstance(asked, TextOutput) else asked


def _guaranteed(request: ResponsesRequest) -> Mapping[str, object] | None:
    """The schema this request asks to be *guaranteed* rather than checked, or `None` for
    every other shape. Under it decoding is constrained, so there is no answer that violates
    the schema; without it the schema is a turn in the prompt and the answer is measured
    against it once the generation is spent."""
    wanted = _wanted(request)
    return wanted.definition if isinstance(wanted, SchemaOutput) and wanted.strict else None


PROFILE_ONLY: Final = frozenset({"reasoning_budget", "reasoning_effort"})
"""Knobs the profile spells in the engine's vocabulary rather than an OpenAI dialect's, so
they are read off the profile where they are used instead of copied onto the request.

`reasoning_budget` is here because neither OpenAI route has a field for it at all and
inventing one would be a field only this server answers. `reasoning_effort` is here because
the two vocabularies differ where it counts: the profile can say `on`, which is thinking
with no rung named, and neither route has a spelling for that — copied across, it would land
in a field typed for OpenAI's words and be read by `effort_of` as none of them.

Both routes share the set, because both refuse for the same reasons."""


def _preset(request: ResponsesRequest, sampling: Sampling) -> ResponsesRequest:
    """The preset — the profile, over the sampling defaults the checkpoint declares — fills
    the knobs the client left out, and only those. Which ones were left
    out is `model_fields_set` — the dialect's defaults are values like any other, so an unset
    field cannot be told from an explicit one by its value."""
    filled = {
        knob: value
        for knob, value in sampling.model_dump(exclude_none=True).items()
        if knob not in request.model_fields_set and knob not in PROFILE_ONLY
    }
    return request.model_copy(update=filled)


# The same invariant `chat/completions` asserts, for the same reason: `model_copy(update=...)`
# writes the keys straight into the instance, so a knob the profile grows and this route does
# not have would be set on the request and read by nobody.
assert set(Sampling.model_fields) - PROFILE_ONLY <= set(ResponsesRequest.model_fields)


def _given_part(part: ContentPart | InputImage) -> TextPart | ImagePart:
    """One part of an input item, as the conversation carries it. Named apart from `_part`
    below, which is this dialect's *output* part and the other direction entirely."""
    if isinstance(part, ContentPart):
        return {"type": "text", "text": part.text}
    return inline_image(part.image_url)


def _turn(item: MessageItem) -> ToolTurn:
    """`developer` is the dialect's newer name for a system turn, and the template knows
    only the older one. The text parts of a content list concatenate: what separates them is a
    boundary in the client's own structure, not text the model should read."""
    content = (
        item.content
        if isinstance(item.content, str)
        else content_of([_given_part(part) for part in item.content])
    )
    return {"role": "system" if item.role == "developer" else item.role, "content": content}


def _given(input: str | list[InputItem]) -> tuple[ToolTurn, ...]:
    """What the client sent, as turns: a bare string is the user message it stands for.

    A call is an item of its own here and a key of the assistant's turn in every template, so
    consecutive calls fold into the one turn that made them — two turns would tell the model
    it answered twice.
    """
    if isinstance(input, str):
        return ({"role": "user", "content": input},)
    turns: list[ToolTurn] = []
    for item in input:
        match item:
            case MessageItem():
                turns.append(_turn(item))
            case FunctionCallItem():
                call: dict[str, object] = {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
                previous = turns[-1] if turns else None
                if previous is not None and previous["role"] == "assistant":
                    # Whether or not it already carries calls: the canonical replay is
                    # `input + response.output`, and a generation that wrote text *and*
                    # called something is `[message, function_call]` — one turn of the model,
                    # which two assistant turns in the prompt would tell it was two.
                    previous["tool_calls"] = [*previous.get("tool_calls", []), call]
                else:
                    turns.append({"role": "assistant", "content": "", "tool_calls": [call]})
            case FunctionOutputItem():
                turns.append({"role": "tool", "content": item.output, "tool_call_id": item.call_id})
    return tuple(turns)


def declared(
    name: str, description: str | None, parameters: Mapping[str, object] | None
) -> Mapping[str, object]:
    """One function offered to the model, in the nested envelope every template reads —
    `tool.function.name` — whatever shape the dialect spelled it in.

    The keys go in in this order because a template that renders the entry with `tojson`
    writes them in it: the same function declared through two dialects has to reach the model
    as the same characters, or the two prompts are two prompts. The schema itself travels
    untouched — what a client declared is what the model reads, and validating a call against
    it is the client's own business.
    """
    function: dict[str, object] = {"name": name}
    if description is not None:
        function["description"] = description
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _tools(request: ResponsesRequest) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: "none"` is honoured where it can be honoured: the tools never enter the
    prompt, so the model has nothing to call rather than an instruction not to."""
    if request.tools is None or request.tool_choice == "none":
        return ()
    return tuple(declared(tool.name, tool.description, tool.parameters) for tool in request.tools)


def _prefixed(
    given: tuple[ToolTurn, ...], instructions: str | None, system_prompt: str | None
) -> tuple[ToolTurn, ...]:
    """The profile's system prompt goes in only when nothing else claimed the place — the
    same precedence the chat route follows, and what keeps the template from rendering two
    system turns for the model to pick between."""
    if instructions is not None:
        given = ({"role": "system", "content": instructions}, *given)
    if system_prompt is None or any(turn["role"] == "system" for turn in given):
        return given
    return ({"role": "system", "content": system_prompt}, *given)


def openai_envelope(message: str, code: str, kind: str = "invalid_request_error") -> dict[str, str]:
    """The body of this dialect's error, apart from the response because a stream carries it
    too: the SDK raises on any frame whose JSON has a truthy `error`, and that is the only way
    to fail a generation that already answered 200."""
    return {"message": message, "type": kind, "code": code}


def openai_error(
    status: int, message: str, code: str, kind: str = "invalid_request_error"
) -> JSONResponse:
    """The OpenAI envelope. Exported because it is the dialect's, not this route's: `app`'s
    validation handler and `auth`'s refusal both dispatch by route prefix, and each needs the
    encoder of whichever dialect the request was speaking."""
    return JSONResponse(status_code=status, content={"error": openai_envelope(message, code, kind)})


def _part(text: str) -> dict[str, object]:
    return {"type": "output_text", "text": text, "annotations": []}


def _message(message_id: str, text: str | None) -> dict[str, object]:
    """The text output item of a generation. `None` is the item the stream opens with —
    announced before any text exists, which is what gives the deltas an item to attach to."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "in_progress" if text is None else "completed",
        "content": [] if text is None else [_part(text)],
    }


def _call_item(item_id: str, call_id: str, name: str, arguments: str | None) -> dict[str, object]:
    """One call, as an output item of its own — which is this dialect's shape and not
    `chat/completions`'s list on the message. `None` is the item the stream opens with: the
    arguments arrive in the delta that follows, whole."""
    return {
        "id": item_id,
        "type": "function_call",
        "status": "in_progress" if arguments is None else "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments or "",
    }


def _items(made: tuple[ToolCall, ...]) -> tuple[tuple[str, str, ToolCall], ...]:
    """The two ids each call needs, drawn before the first frame goes out: the item's, which
    the stream repeats on every frame about it, and the one the client answers with."""
    return tuple((f"fc_{uuid.uuid4().hex}", f"call_{uuid.uuid4().hex}", call) for call in made)


def _usage(meter: Meter) -> dict[str, object]:
    """`cache_write_tokens` stays at zero and is what it says: the trie is filled out of the
    forward this turn was going to run anyway, so there is nothing the client was charged
    for storing."""
    return {
        "input_tokens": meter.prompt_tokens,
        "input_tokens_details": {
            "cached_tokens": meter.reused_tokens,
            "cache_write_tokens": 0,
        },
        "output_tokens": meter.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": meter.prompt_tokens + meter.completion_tokens,
    }


def _response(
    request_id: str,
    created: int,
    request: ResponsesRequest,
    output: list[dict[str, object]],
    status: str,
    meter: Meter | None,
    error: str | None = None,
) -> dict[str, object]:
    """The whole resource, which the stream carries twice: once empty and in progress at
    `response.created`, once complete at the end. The SDK builds its snapshot from the first
    and replaces it with the second, so both are the same shape.

    A generation the budget cut is `incomplete` and says why: `completed` with the text cut
    mid-sentence is what an agent loop reads as the final answer, and nothing else in the body
    tells it otherwise."""
    cut = (
        status == "completed"
        and meter is not None
        and meter.completion_tokens >= request.max_output_tokens
    )
    return {
        "id": request_id,
        "object": "response",
        "created_at": created,
        "status": "incomplete" if cut else status,
        "model": request.model,
        "output": output,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "parallel_tool_calls": False,
        "temperature": request.temperature,
        "tool_choice": request.tool_choice,
        "tools": [tool.model_dump(exclude_none=True) for tool in request.tools or ()],
        "top_p": request.top_p,
        "metadata": {},
        "incomplete_details": {"reason": "max_output_tokens"} if cut else None,
        "usage": None if meter is None else _usage(meter),
        "error": None if error is None else {"code": "server_error", "message": error},
    }


def _event(name: str, sequence: int, fields: dict[str, object]) -> str:
    """`event:` as well as the `type` inside the payload: the SDK reads the payload, and a
    client reading the event name off the line — which is what the wire format is for — would
    otherwise see every frame as anonymous."""
    payload: dict[str, object] = {"type": name, "sequence_number": sequence, **fields}
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


async def _drain(job: Job) -> list[Segment]:
    pieces: list[Segment] = []
    while (piece := await job.chunks.get()) is not None:
        pieces.append(piece)
    return pieces


async def _events(
    job: Job,
    request_id: str,
    message_id: str,
    created: int,
    request: ResponsesRequest,
    calls: Calls | None,
    checked: Checked | None,
) -> AsyncIterator[str]:
    sequence = count()
    pieces: list[str] = []
    started = False

    def deltas(content: str) -> Iterator[str]:
        """The item and its content part are announced on the first text there is, not before:
        a generation that only called something has no message item at all, and one announced
        empty would put a blank assistant turn in the client's transcript."""
        nonlocal started
        if not started:
            started = True
            yield _event(
                "response.output_item.added",
                next(sequence),
                {"output_index": 0, "item": _message(message_id, None)},
            )
            yield _event(
                "response.content_part.added",
                next(sequence),
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": message_id,
                    "part": _part(""),
                },
            )
        pieces.append(content)
        yield _event(
            "response.output_text.delta",
            next(sequence),
            {
                "output_index": 0,
                "content_index": 0,
                "item_id": message_id,
                "delta": content,
                "logprobs": [],
            },
        )

    try:
        opened = _response(request_id, created, request, [], "in_progress", None)
        yield _event("response.created", next(sequence), {"response": opened})
        while True:
            try:
                piece = await asyncio.wait_for(job.chunks.get(), _KEEP_ALIVE_SECONDS)
            except TimeoutError:
                # Keeps the connection warm through a long prefill.
                yield ": keep-alive\n\n"
                continue
            if piece is None:
                break
            # No family, nothing that could read a channel back: every segment is text the
            # client asked for, whichever one the model wrote it on.
            if content := (piece.text if calls is None else calls.push(piece)):
                for frame in deltas(content):
                    yield frame
        made: tuple[ToolCall, ...] = ()
        if calls is not None:
            tail, made = calls.finish()
            if tail:
                for frame in deltas(tail):
                    yield frame
        text = "".join(pieces)
        if job.error is not None:
            # The text that did arrive stays in the item: what failed is the rest of the
            # generation, and a client that already rendered those deltas is not told they
            # never happened. `response.completed` is the frame the SDK accumulates into a
            # final response, so a failure must not wear it.
            broke = _response(
                request_id,
                created,
                request,
                [_message(message_id, text)] if started else [],
                "failed",
                job.meter,
                job.error,
            )
            yield _event("response.failed", next(sequence), {"response": broke})
            return
        if checked is not None and not made:
            try:
                document(text, checked)
            except (MalformedJSON, SchemaViolation) as violation:
                # A stream cannot take back the text it already handed out, so the violation
                # travels the way a generation that died travels: `response.failed`, which is
                # the one frame the SDK's accumulator refuses to build a final response out
                # of. `response.completed` here would hand the client a document it believes
                # was checked. The deltas stay as they were sent — what failed is the answer.
                reason, _ = failed(violation, checked.attempts)
                rejected = _response(
                    request_id,
                    created,
                    request,
                    [_message(message_id, text)] if started else [],
                    "failed",
                    job.meter,
                    reason,
                )
                yield _event("response.failed", next(sequence), {"response": rejected})
                return
        output: list[dict[str, object]] = []
        if started:
            yield _event(
                "response.output_text.done",
                next(sequence),
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": message_id,
                    "text": text,
                    "logprobs": [],
                },
            )
            yield _event(
                "response.content_part.done",
                next(sequence),
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": message_id,
                    "part": _part(text),
                },
            )
            yield _event(
                "response.output_item.done",
                next(sequence),
                {"output_index": 0, "item": _message(message_id, text)},
            )
            output.append(_message(message_id, text))
        for item_id, call_id, call in _items(made):
            # The arguments in one delta and whole, which is A11's decision: the SDK's
            # accumulator concatenates them into the item it was told about above, and a
            # single fragment is the valid degenerate case of that.
            index = len(output)
            arguments = json.dumps(call.arguments)
            yield _event(
                "response.output_item.added",
                next(sequence),
                {"output_index": index, "item": _call_item(item_id, call_id, call.name, None)},
            )
            yield _event(
                "response.function_call_arguments.delta",
                next(sequence),
                {"output_index": index, "item_id": item_id, "delta": arguments},
            )
            yield _event(
                "response.function_call_arguments.done",
                next(sequence),
                {
                    "output_index": index,
                    "item_id": item_id,
                    "name": call.name,
                    "arguments": arguments,
                },
            )
            item = _call_item(item_id, call_id, call.name, arguments)
            yield _event(
                "response.output_item.done", next(sequence), {"output_index": index, "item": item}
            )
            output.append(item)
        done = _response(request_id, created, request, output, "completed", job.meter)
        yield _event("response.completed", next(sequence), {"response": done})
    finally:
        job.cancel()


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


@router.post("/api/openai/v1/responses", response_model=None)
async def respond(
    request: ResponsesRequest, engine: EngineDep, store: StoreDep
) -> JSONResponse | StreamingResponse:
    if request.store:
        return openai_error(
            400,
            "store is not supported: this server keeps no responses, so an id it handed "
            "back would name nothing",
            "store_unsupported",
        )
    if any(tool.strict for tool in request.tools or ()):
        return openai_error(
            400,
            "strict on a tool is not supported: what a grammar constrains here is the whole "
            "answer (text.format), not the arguments of a call, so an argument that violates "
            "the schema would come back as one that was checked against it",
            "strict_unsupported",
        )
    if _guaranteed(request) is not None and _tools(request):
        return openai_error(
            400,
            "a strict text.format and tools cannot both be honoured: the grammar constrains "
            "decoding to the schema from the first token, so the model cannot write a call "
            "however it is offered. Send the same format without strict, or offer no tools.",
            "strict_with_tools",
        )
    try:
        given = _given(request.input)
    except UnreadableImage as unreadable:
        return openai_error(400, str(unreadable), "invalid_image")
    if not any(turn["content"] or turn.get("tool_calls") for turn in given):
        return openai_error(400, "input must contain non-empty text", "empty_input")

    model_id, profile = profiles.resolve(store, request.model)
    preset = profiles.preset(model_id, profile)
    asked = _preset(request, preset)
    turns = _prefixed(
        given, request.instructions, None if profile is None else profile.system_prompt
    )
    tools = _tools(request)
    strict = _guaranteed(request)
    wanted = _wanted(request)
    # At most one of the two levels: with `strict` the grammar makes a violation unreachable
    # and there is nothing left to check afterwards.
    checked = (
        None
        if strict is not None or wanted is None
        else Checked(wanted.definition if isinstance(wanted, SchemaOutput) else None, 1)
    )
    if checked is not None:
        turns = (*turns, instruction(checked.schema))
    # The conversation goes to the model as a conversation: what turns it into a prompt is
    # the checkpoint's own chat template, and a model that ships none says so below.
    reasoning = request.reasoning
    effort = effort_of(None if reasoning is None else reasoning.effort, preset.reasoning_effort)
    conversation = Chat(turns, tools=tools, reasoning_effort=effort)
    request_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        constrained = None if strict is None else await engine.constrain(model_id, strict)
        # A name that does not resolve to a checkpoint and one whose load fails are the same
        # answer to the client: this model is not available here.
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
        # "grammar error" is not, and what the client does with it is send the same schema
        # without `strict` and have the answer checked instead of guaranteed.
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
    # envelope no parser answers to costs the client the text and gives back no call.
    parser = parser_of(job.model) if tools else None
    family = None if parser is None else parser.tools
    calls = None if family is None else Calls(family)

    if request.stream:
        # `asked` and not `request` is what the answer echoes: the knobs a profile filled are
        # the ones that generated the text, and a client reading its own back learns nothing.
        return StreamingResponse(
            _events(job, request_id, message_id, created, asked, calls, checked),
            media_type="text/event-stream",
        )

    try:
        pieces = await _drain(job)
    finally:
        job.cancel()
    if job.error is not None:
        return openai_error(500, job.error, "generation_failed", kind="server_error")
    content = "".join(piece.text for piece in pieces)
    made: tuple[ToolCall, ...] = ()
    if calls is not None:
        held = "".join(calls.push(piece) for piece in pieces)
        tail, made = calls.finish()
        content = held + tail
    # A turn that called something answered with a call and not with a document: the format is
    # about the answer, and this turn has none to check.
    if checked is not None and not made:
        try:
            value = document(content, checked)
        except (MalformedJSON, SchemaViolation) as violation:
            reason, code = failed(violation, checked.attempts)
            # 422 and not a 5xx: the SDKs retry a 5xx on their own, and a whole generation
            # bought behind the client's back is the cost this level exists to make visible.
            return openai_error(422, reason, code, kind="server_error")
        # The document and not the text around it: the client asked for a JSON value, and
        # prose or a fence with one inside it is an answer `json.loads` refuses. What goes
        # back is what was validated.
        content = json.dumps(value, ensure_ascii=False)
    # A turn that only called something has no message item: an empty one is an assistant
    # that answered with nothing, which is not what happened.
    written = not made or bool(content)
    output: list[dict[str, object]] = [_message(message_id, content)] if written else []
    output += [
        _call_item(item_id, call_id, call.name, json.dumps(call.arguments))
        for item_id, call_id, call in _items(made)
    ]
    return JSONResponse(
        content=_response(request_id, created, asked, output, "completed", job.meter)
    )
