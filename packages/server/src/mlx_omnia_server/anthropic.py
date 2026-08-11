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

Reasoning is a **block**, not text. `<think>` reaches this module already labelled — the
engine's segmenter cut it — so it leaves as the `thinking` block the dialect has for it,
which is what keeps a client from drawing the model's deliberation as the answer. `thinking`
on the way in says whether the template is asked for it (`type`), how long it may go on
(`budget_tokens`) and whether the client is shown it (`display`); `thinking` blocks on the
way back in are read and dropped,
because no template renders a previous turn's reasoning and a conversation that used it must
still be replayable.

`stop_sequences` are honoured over the text. The engine's `stop` is a set of token ids and
cannot express a string, so what ends the turn is the sequence appearing in what the model
wrote: the answer is cut before it, the generation is cancelled, and `stop_reason` says
`stop_sequence` with the match beside it. Held like a marker on the way out (`Halt`) — a
piece that ends on half a sequence must not go out before the piece that decides what it was.

Two routes stand beside `/messages`: `/messages/count_tokens`, which answers about the
rendered prompt with the checkpoint's own tokenizer, and `GET /models/{id}`, which is the
listing's single entry. Neither generates anything, and the first is the only place in the
dialect where a prompt exists on this side of the queue.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import count
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlx_omnia import (
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
from mlx_omnia.chat import Effort, parser_of, template_of
from mlx_omnia.generate import Constraint, Meter
from mlx_omnia.grammar import GrammarRefused
from mlx_omnia.language import tokenizer_of
from mlx_omnia.parsers import Segment, ToolCall, unmarked
from mlx_omnia_server import profiles
from mlx_omnia_server.engine import Engine, Job, NotConstrainable, NotQuantizable
from mlx_omnia_server.profiles import Sampling, StoreDep
from mlx_omnia_server.responses import (
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


class ThinkingBlock(BaseModel):
    """The reasoning of an earlier turn, replayed. It is read and dropped: no chat template
    in circulation renders a previous turn's thinking — the ones that write `<think>` at all
    write it for the turn being generated — so a block kept here would enter the prompt as
    prose the model wrote about a question it has already answered. Accepting it is what
    matters: the dialect asks a client to send the assistant's blocks back unchanged, and a
    conversation that used thinking is unreplayable if the block is refused."""

    type: Literal["thinking"]
    thinking: str = ""
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    """`thinking`'s other spelling, for the same reason and with the same fate. Nothing here
    encrypts a block, so this only ever arrives from a conversation that began somewhere
    else — and it is still a turn the client must be able to replay."""

    type: Literal["redacted_thinking"]
    data: str = ""


type Block = (
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock | RedactedThinkingBlock
)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    """`system` is a turn *and* a field in this dialect, and the two mean different things:
    the field opens the conversation, the role appends an operator instruction mid-way —
    which is what a client sends to steer a running session without editing the prompt it
    has already cached. The position rules upstream (never first, never between two
    assistant turns) are not enforced here: what they protect is a cache this server does
    not keep, and a template renders the turn wherever it sits."""
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
    object carries upstream: it is accepted and changes nothing, because it asks for a dial
    this server does not have — how long a checkpoint thinks is the checkpoint's, not a
    knob. It is accepted rather than refused for the reason `metadata` is (see
    `MessagesRequest`): a client that sends it on every request would otherwise never reach
    this dialect at all, and what it asks for is spend, not an answer of another kind."""

    model_config = ConfigDict(extra="forbid")

    format: JsonFormat | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class Thinking(BaseModel):
    """Whether the checkpoint reasons before it answers, how long, and whether the client is
    shown it.

    `adaptive` and `enabled` both reach the template as thinking on and `disabled` as off —
    a switch and never a rung, because this dialect names states and not levels, and a
    request that asked for neither `low` nor `high` is not owed one in its prompt. A client
    that wants a rung names a profile, which is where the effort lives whole.

    What tells the two on-modes apart upstream is `budget_tokens`, and here it is the same
    thing: it is the number of ids the reasoning block may spend, and the block is ended by
    feeding its closer once they are gone — so what comes back is a finished thought and an
    answer after it, not a generation cut mid-sentence. `adaptive` without one leaves the
    length to the checkpoint, which is what the mode says. With `disabled` it is refused
    rather than dropped: there is no block for it to bound.

    `display` is what the *client* is shown, not what the model does: `omitted` leaves the
    thinking blocks in the answer with their text empty, which is upstream's default and is
    exactly what a client that never wanted the reasoning asked for. `summarized` returns
    the reasoning itself — there is no summarizer here, and the raw block is more than the
    summary, not less."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["adaptive", "enabled", "disabled"]
    budget_tokens: int | None = Field(default=None, ge=0)
    display: Literal["omitted", "summarized"] = "omitted"

    @model_validator(mode="after")
    def _budget_needs_a_block(self) -> "Thinking":
        if self.type == "disabled" and self.budget_tokens is not None:
            raise ValueError(
                "thinking.budget_tokens has nothing to bound when thinking is disabled"
            )
        return self


class ClearThinking(BaseModel):
    """The one context edit this server can honour, and it honours it by construction: no
    thinking block ever enters a prompt here (see `ThinkingBlock`), so a conversation with
    this edit applied and one without render the same. Its keys are dropped rather than
    read for the same reason — whatever this asks to clear is already absent."""

    type: Literal["clear_thinking_20251015"]


class ContextManagement(BaseModel):
    """`clear_tool_uses_20250919` and `compact_20260112` are refused by the discriminator:
    both are decided by token thresholds against the rendered prompt, and rendering happens
    on the far side of the queue, inside the engine. Answering them by dropping the edit
    would hand the model a conversation the client asked to have shortened."""

    model_config = ConfigDict(extra="forbid")

    edits: list[Annotated[ClearThinking, Field(discriminator="type")]] = []


class Metadata(BaseModel):
    """Who the request is for, upstream. Accepted and ignored: nothing here bills, meters or
    rate-limits per user, so the field asks for nothing this server would do differently.
    Dropped rather than refused because it says nothing about the answer — a request refused
    over it is a client that cannot talk to this dialect at all."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None


class Conversation(BaseModel):
    """What the two routes of this dialect have in common: the turns, and everything that
    decides what the prompt says. `count_tokens` answers about exactly this — the same body
    minus what is about the generation — so it is one class and not two that drift."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[Message]
    system: str | list[TextBlock] | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: Thinking | None = None
    context_management: ContextManagement | None = None


class CountRequest(Conversation):
    """`POST /messages/count_tokens`'s body: the conversation and nothing else. `max_tokens`
    is absent because nothing is generated, and the sampling knobs with it."""


class MessagesRequest(Conversation):
    """Unknown fields are refused rather than dropped, and the ones that get here are named
    by the refusal. What the dialect carries upstream and this server does not implement —
    `container`, `mcp_servers`, `fallbacks`, `service_tier` — refuses that way: each names
    something that runs somewhere else, and a client told it was honoured was told the
    answer came from a machine it did not.

    The line between a field accepted-and-ignored and one refused is what it would change:
    `metadata` and `output_config.effort` ask for spend and accounting, which is nobody's
    answer, so they are accepted; anything that would change what the model is asked, or
    what the answer means, is refused with the name of the field in the message."""

    max_tokens: int = Field(gt=0)
    """Required, which is the dialect's own decision and not an omission: Anthropic has no
    default budget, and every SDK sends one."""
    output_config: OutputConfig | None = None
    """The schema this answer is decoded under, when the client sent one. There is no checked
    flavour of it in this dialect and none is invented here: the schema is compiled into a
    grammar and the ids that would break it are at -inf before the draw, or the request is
    refused with the compiler's own words."""
    metadata: Metadata | None = None
    stop_sequences: list[str] = []
    """The strings this generation ends on. Honoured over the text and not over the ids: the
    engine's `stop` is a set of token ids and cannot express a string, so what ends the turn
    here is the sequence appearing in what the model wrote — the answer is cut before it, the
    generation is cancelled, and `stop_reason` is `stop_sequence` with the match beside it.
    Only the answer is matched: the reasoning is another channel, and a client's sequence is
    about the text it will read."""
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


class Halt:
    """The sequences this request ends on, over the text as it arrives.

    Held like the engine's own segmenter holds a marker, and for the same reason: a piece
    ending in `<` when the client asked to stop on `<end>` must not go out, because the next
    piece decides whether those characters are the answer or the sequence that ends it. What
    is held never exceeds the longest sequence minus one character, so a generation that
    writes none streams with the boundaries it would have had.

    The earliest match wins when two sequences match in the same push — it is the one the
    model reached first, and it is what `stop_sequence` reports.
    """

    def __init__(self, sequences: Sequence[str]) -> None:
        self._sequences = tuple(sequence for sequence in sequences if sequence)
        self._held = ""
        self.matched: str | None = None

    def push(self, text: str) -> str:
        """What may go out now: everything before a match, and never a prefix of one. Empty
        once something has matched — the generation is over and the rest is not the answer."""
        if self.matched is not None:
            return ""
        if not self._sequences:
            return text
        self._held += text
        found = [(self._held.find(s), s) for s in self._sequences]
        hits = sorted((at, s) for at, s in found if at != -1)
        if hits:
            at, self.matched = hits[0]
            out, self._held = self._held[:at], ""
            return out
        keep = max(
            (
                size
                for sequence in self._sequences
                for size in range(1, len(sequence))
                if self._held.endswith(sequence[:size])
            ),
            default=0,
        )
        at = len(self._held) - keep
        out, self._held = self._held[:at], self._held[at:]
        return out

    def finish(self) -> str:
        """What was held when the generation ended without matching — the tail of the answer,
        which was only ever held because it could still have become a sequence."""
        out, self._held = self._held, ""
        return "" if self.matched is not None else out


class Blocks:
    """The answer as blocks, in the order the model wrote them: one block per run of a
    channel, and consecutive pieces of the same channel in the same block.

    The stream has no choice about this — a frame goes out when it arrives — so the answer
    assembled here has none either: a message that gathered all the reasoning into one block
    at the front would disagree with the stream about the same generation.
    """

    def __init__(self) -> None:
        self._runs: list[tuple[str, list[str]]] = []

    def wrote(self, kind: str, text: str) -> None:
        if not text:
            return
        if self._runs and self._runs[-1][0] == kind:
            self._runs[-1][1].append(text)
            return
        self._runs.append((kind, [text]))

    def blocks(self, shown: bool) -> list[dict[str, object]]:
        """`shown` is `thinking.display`: the block goes out either way, because a client
        that did not want to read the reasoning did not ask to be told there was none."""
        return [
            _thought("".join(parts) if shown else "")
            if kind == "thinking"
            else {"type": "text", "text": "".join(parts)}
            for kind, parts in self._runs
        ]


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


def _tools(request: Conversation) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: {"type": "none"}` is honoured where it can be honoured: the tools never
    enter the prompt, so the model has nothing to call rather than an instruction not to."""
    choice = request.tool_choice
    if request.tools is None or (choice is not None and choice.type == "none"):
        return ()
    return tuple(declared(tool.name, tool.description, tool.input_schema) for tool in request.tools)


def _thinks(request: Conversation, preset: Effort | None) -> Effort:
    """What the template is told about thinking.

    A request that names `thinking` throws the switch and nothing more: this dialect has
    states and no rungs, so `enabled` and `adaptive` are `on` and `disabled` is `off`. One
    that names none falls to the profile, and then to `auto` — the template's own default
    and not `off`, because a checkpoint that reasons by design is not asked to stop over a
    field a client left out."""
    thinking = request.thinking
    if thinking is None:
        return "auto" if preset is None else preset
    return "off" if thinking.type == "disabled" else "on"


def _shown(request: Conversation) -> bool:
    """Whether the reasoning reaches the client as text. A request that named no `thinking`
    gets it: the block is what the model wrote, this server has no other channel for it, and
    dropping it silently is how the reasoning of a whole turn goes missing. A request that
    named one gets what it asked for, `omitted` included — that is upstream's default and
    the client chose it."""
    thinking = request.thinking
    return True if thinking is None else thinking.display == "summarized"


def _conversation(request: Conversation, preset: str | None, effort: Effort | None = None) -> Chat:
    """The translation this dialect exists for: `system` is a field on the way in and the
    first turn on the way out. The profile's prompt fills it only when the request left it
    out — the same precedence the sampling knobs below follow, `effort` included."""
    system = _text(request.system) if request.system is not None else preset
    turns: list[ToolTurn] = [] if system is None else [{"role": "system", "content": system}]
    for message in request.messages:
        turns += _turns(message)
    return Chat(tuple(turns), tools=_tools(request), reasoning_effort=_thinks(request, effort))


def _options(
    request: MessagesRequest, sampling: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """The preset — the profile, over the sampling defaults the checkpoint declares — fills
    the knobs the client left out, and only those: a request that names
    a temperature means it. Which ones it left out is `model_fields_set` — the dialect's
    defaults are values like any other, so an unset field cannot be told from an explicit
    one by its value. `min_p`, `repetition_penalty` and `seed` have no field here at all,
    so a profile is the only thing that can set them.

    The reasoning budget has a field here — `thinking.budget_tokens` — and follows the same
    rule: what the request names wins, and the profile fills the silence. `adaptive` without
    a budget is silence, which is what the mode means.

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
    thinking = request.thinking
    budget = None if thinking is None else thinking.budget_tokens
    if budget is None:
        budget = sampling.reasoning_budget
    if heat == 0.0:
        # The deterministic end of the dial: no distribution is left to draw from, and
        # dividing by it would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=request.max_tokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=budget,
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
        reasoning_budget=budget,
    )


def _usage(meter: Meter, *, output: int) -> dict[str, int]:
    """`output` is a parameter because the two callers disagree about it and neither is
    wrong: the frame that opens a stream is written before a token exists, and the answer is
    written after the last one.

    The three input fields are disjoint here, which is this dialect's arithmetic and not the
    OpenAI one: a client adds them up to get the prompt, so `input_tokens` is what was
    actually read this turn and the reuse is beside it. `cache_creation_input_tokens` stays
    at zero because it is — the trie is filled out of the forward this turn already ran, and
    there is no separate write to charge for."""
    return {
        "input_tokens": meter.prompt_tokens - meter.reused_tokens,
        "cache_read_input_tokens": meter.reused_tokens,
        "cache_creation_input_tokens": 0,
        "output_tokens": output,
    }


def _message(
    message_id: str,
    model: str,
    content: list[dict[str, object]],
    stop_reason: str | None,
    usage: Mapping[str, int],
    stop_sequence: str | None = None,
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
        "stop_sequence": stop_sequence,
        "usage": usage,
    }


def _use(block_id: str, name: str, input: Mapping[str, object]) -> dict[str, object]:
    """A call, as a block of the answer. `input` is empty in the block that opens one on a
    stream: what fills it is the `input_json_delta` that follows."""
    return {"type": "tool_use", "id": block_id, "name": name, "input": input}


def _thought(text: str) -> dict[str, object]:
    """The reasoning, as a block of the answer. `signature` is empty and present: upstream
    signs the block so it can verify its own on the way back, and this dialect's reader
    (`ThinkingBlock`) verifies nothing — what the field owes a client is a place to put what
    it must replay."""
    return {"type": "thinking", "thinking": text, "signature": ""}


def _stop_reason(job: Job, max_tokens: int, called: bool, halted: bool) -> str:
    """A call, a sequence the client asked to stop on, the budget, or the model's own end
    token. `tool_use` outranks the rest: what is reported is only ever a call read whole, so a
    client that gets one can execute it, and `max_tokens` beside an executable call sends it
    to render a truncation instead. `stop_sequence` outranks the budget for the same shape of
    reason — a turn cut at the client's own sequence is that turn ending, whatever the meter
    reached on the token that carried it."""
    if called:
        return "tool_use"
    if halted:
        return "stop_sequence"
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
    job: Job,
    message_id: str,
    model: str,
    max_tokens: int,
    calls: Calls | None,
    shown: bool,
    halt: Halt,
) -> AsyncIterator[str]:
    """`message_start` waits for the first piece instead of going out at once, because it
    carries `input_tokens` and the prompt is only counted once the checkpoint's own template
    has rendered it and the model has tokenized it — which happens on the other side of the
    queue. What holds the connection open until then is the ping the dialect has for it.

    Blocks are opened one at a time and closed when the channel turns: reasoning and the
    answer are two content blocks of one message here, in the order the model wrote them, and
    a delta whose type does not match the block it lands in is an accumulator that raises."""
    blocks = count()
    index: int | None = None
    kind: str | None = None

    def closed() -> Iterator[str]:
        nonlocal index, kind
        if index is not None:
            yield _frame("content_block_stop", {"type": "content_block_stop", "index": index})
        index, kind = None, None

    def opened(wanted: str, block: Mapping[str, object]) -> Iterator[str]:
        """The block a channel writes into, opened on the first piece of it and not before: a
        generation that only called something has no text block, and one opened empty is an
        assistant that answered `""` before it called."""
        nonlocal index, kind
        if kind == wanted:
            return
        yield from closed()
        index, kind = next(blocks), wanted
        yield _frame(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": block},
        )

    def deltas(content: str) -> Iterator[str]:
        yield from opened("text", {"type": "text", "text": ""})
        assert index is not None
        yield _delta(index, content)

    def reasoned(content: str) -> Iterator[str]:
        """The block goes out whether or not its text does: `display: "omitted"` is a client
        that does not want to read the reasoning, not one that wants to be told the model
        never reasoned."""
        yield from opened("thinking", _thought(""))
        assert index is not None
        if shown and content:
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": content},
                },
            )

    def emitted(piece: Segment) -> Iterator[str]:
        if piece.channel == "reasoning":
            yield from reasoned(unmarked(piece.text))
            return
        # No family, nothing that could read a channel back: every segment is text the client
        # asked for, whichever one the model wrote it on.
        content = piece.text if calls is None else calls.push(piece)
        if content := halt.push(content):
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

        usage = _usage(job.meter, output=0)
        yield _frame(
            "message_start",
            {"type": "message_start", "message": _message(message_id, model, [], None, usage)},
        )
        if first is not None:
            for frame in emitted(first):
                yield frame
        if halt.matched is None:
            async for piece in pieces:
                if piece is None:
                    yield _frame("ping", {"type": "ping"})
                    continue
                for frame in emitted(piece):
                    yield frame
                if halt.matched is not None:
                    # The turn is over at the client's own word: what the model writes after
                    # the sequence is not the answer, and the generation is cancelled by the
                    # `finally` below rather than drained for text nobody will read.
                    break

        made: tuple[ToolCall, ...] = ()
        if calls is not None and halt.matched is None:
            tail, made = calls.finish()
            if tail := halt.push(tail):
                for frame in deltas(tail):
                    yield frame
        if held := halt.finish():
            for frame in deltas(held):
                yield frame
        if job.error is not None and halt.matched is None:
            # The status is long gone — the response opened 200 the moment the first frame
            # went out — so the dialect's own event is the only place left to say it.
            yield _frame(
                "error", {"type": "error", "error": {"type": "api_error", "message": job.error}}
            )
            return
        for frame in closed():
            yield frame
        for call in made:
            # The call whole, in the one delta its block carries: the SDK's accumulator
            # concatenates `partial_json` and parses what it has at every step, so a single
            # fragment is the valid degenerate case of that. A11's decision.
            at = next(blocks)
            yield _frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": at,
                    "content_block": _use(f"toolu_{uuid.uuid4().hex}", call.name, {}),
                },
            )
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": at,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(call.arguments),
                    },
                },
            )
            yield _frame("content_block_stop", {"type": "content_block_stop", "index": at})
        yield _frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _stop_reason(
                        job, max_tokens, bool(made), halt.matched is not None
                    ),
                    "stop_sequence": halt.matched,
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


@router.get("/api/anthropic/v1/models/{model_id:path}")
def model(model_id: str, store: StoreDep) -> JSONResponse:
    """One entry of the listing above, by the name it answers to. `{model_id:path}` because a
    served id carries slashes — `vendor/name`, and `vendor/name:profile` for a profile — and a
    route that stopped at the first one would answer for half a name.

    A name nothing serves is `not_found_error` and not an empty object: the SDK reads this to
    decide whether a model exists, and an id it made up must not come back looking real."""
    if model_id not in profiles.served_ids(store):
        return encode_error(404, "not_found_error", f"model {model_id!r} is not available here")
    return JSONResponse(
        content={
            "id": model_id,
            "type": "model",
            "display_name": model_id,
            "created_at": "1970-01-01T00:00:00Z",
        }
    )


def _pictured(conversation: Chat) -> bool:
    """Whether an image reached the turns, in the shape `content_of` leaves them: characters
    when the turn is text, the parts themselves when one of them is a picture."""
    return any(
        not isinstance(content, str) and any(part["type"] == "image" for part in content)
        for content in (message["content"] for message in conversation.messages)
    )


@router.post("/api/anthropic/v1/messages/count_tokens", response_model=None)
async def count_tokens(request: CountRequest, engine: EngineDep, store: StoreDep) -> JSONResponse:
    """What this conversation costs to send, in the tokens the checkpoint itself counts.

    Only a resident model answers, and a name that is not resident is refused rather than
    loaded — `/admin/models/{id}/tokenize`'s decision, for its reason: the caller is a client
    counting a context window as it fills, and a load is seconds and tens of gigabytes asked
    for on purpose through `PUT /admin/models/{id}/residency`, never as the side effect of a
    count.

    The count is of the rendered prompt and not of the turns: what the tokens are is decided
    by the chat template — the turn markers, the tools written into the system turn, the
    generation prompt at the end — and a sum over the messages would be a number no request
    ever pays. An image is refused instead of skipped: how many tokens a picture becomes is
    the checkpoint's own processor's to say, and this route renders text.

    The encode goes to a thread, like the tokenize route's: the BPE is Python, a full context
    is hundreds of thousands of characters, and the loop it would otherwise sit on is the one
    carrying token streams.
    """
    model_id, profile = profiles.resolve(store, request.model)
    if model_id not in engine.resident:
        return encode_error(
            409,
            "invalid_request_error",
            f"{model_id!r} is not resident: load it before counting tokens with it",
        )
    # The body before the checkpoint: what the client sent is the client's to fix, and hearing
    # about the deployment first sends it looking at the wrong end of the request.
    try:
        conversation = _conversation(
            request,
            None if profile is None else profile.system_prompt,
            None if profile is None else profile.sampling.reasoning_effort,
        )
    except UnreadableImage as unreadable:
        return encode_error(400, "invalid_request_error", str(unreadable))
    if _pictured(conversation):
        return encode_error(
            400,
            "invalid_request_error",
            "an image's tokens are counted by the checkpoint's own processor and not by this "
            "route, which renders the conversation as text: count the turns without it",
        )
    resolved = await engine.resolve(model_id)
    template = template_of(resolved)
    tokenizer = tokenizer_of(resolved)
    if template is None:
        return encode_error(
            400, "invalid_request_error", unsupported_reason(request.model, Chat(()))
        )
    if tokenizer is None:
        return encode_error(400, "invalid_request_error", f"{model_id!r} exposes no tokenizer")
    try:
        prompt = template.render(conversation)
    except Exception as refusal:
        # A template that will not render these turns is a prompt this conversation cannot
        # become, and the client is the only one who can change it — the same 400 the
        # generation route answers when the render fails on the far side of the queue.
        return encode_error(400, "invalid_request_error", str(refusal))
    # Read out in the thread, not after it: `encode` hands the ids over as it makes them.
    def encode() -> list[int]:
        return list(tokenizer.encode(prompt))

    ids = await asyncio.to_thread(encode)
    return JSONResponse(content={"input_tokens": len(ids)})


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
        return encode_error(400, "invalid_request_error", "messages must contain non-empty content")
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
    preset = profiles.preset(model_id, profile)
    try:
        conversation = _conversation(
            request,
            None if profile is None else profile.system_prompt,
            preset.reasoning_effort,
        )
    except UnreadableImage as unreadable:
        return encode_error(400, "invalid_request_error", str(unreadable))
    message_id = f"msg_{uuid.uuid4().hex}"
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        walk = None if schema is None else await engine.constrain(model_id, schema)
        # A name no checkpoint answers to and one whose load fails are the same answer to the
        # client: this model is not available here. A `model:typo` lands here whole — no
        # profile matched, so the name was never split.
        job = await engine.submit(
            model_id,
            conversation,
            replace(
                _options(request, preset, walk),
                speculate=profiles.speculating(store, model_id, profile),
            ),
        )
    except GrammarRefused as refusal:
        # The compiler's own words — `Unimplemented keys: ["uniqueItems"]` is a reason where
        # "grammar error" is not, and it is what tells the client which keyword to drop.
        return encode_error(400, "invalid_request_error", str(refusal))
    except NotConstrainable as refusal:
        return encode_error(400, "invalid_request_error", str(refusal))
    except NotQuantizable as refusal:
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
    parser = parser_of(job.model) if conversation.tools else None
    family = None if parser is None else parser.tools
    calls = None if family is None else Calls(family)

    halt = Halt(request.stop_sequences)
    if request.stream:
        return StreamingResponse(
            _events(
                job,
                message_id,
                request.model,
                request.max_tokens,
                calls,
                _shown(request),
                halt,
            ),
            media_type="text/event-stream",
        )

    # The two channels part here for the reason the stream parts them — the reasoning is a
    # block of its own in this dialect, and the sequence, the schema and the call are all
    # about the answer — and they part in the order the model wrote them. A message assembled
    # channel by channel instead would disagree with the stream about the same generation:
    # there the blocks can only come out as they arrive, and one answer read two ways is two
    # answers.
    written = Blocks()
    try:
        while (piece := await job.chunks.get()) is not None:
            if piece.channel == "reasoning":
                written.wrote("thinking", unmarked(piece.text))
                continue
            written.wrote("text", halt.push(piece.text if calls is None else calls.push(piece)))
            if halt.matched is not None:
                # The client's sequence, in the text as it arrived: what the model writes
                # after it is not the answer, so it is not waited for either.
                break
    finally:
        job.cancel()
    if job.error is not None and halt.matched is None:
        return encode_error(500, "api_error", job.error)
    made: tuple[ToolCall, ...] = ()
    if calls is not None and halt.matched is None:
        tail, made = calls.finish()
        written.wrote("text", halt.push(tail))
    written.wrote("text", halt.finish())
    # A turn that only called something carries no text block: an empty one is an assistant
    # that answered with nothing, which is not what happened. A turn that wrote nothing at all
    # and called nothing still carries one — that is the answer, and it was empty.
    content: list[dict[str, object]] = written.blocks(_shown(request))
    if not content and not made:
        content = [{"type": "text", "text": ""}]
    content += [_use(f"toolu_{uuid.uuid4().hex}", call.name, call.arguments) for call in made]
    return JSONResponse(
        content=_message(
            message_id,
            request.model,
            content,
            _stop_reason(job, request.max_tokens, bool(made), halt.matched is not None),
            _usage(job.meter, output=job.meter.completion_tokens),
            halt.matched,
        )
    )
