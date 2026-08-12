"""What a dialect is, and the machine that cuts a generation along one.

A dialect is the whole grammar of one checkpoint's output: how it spells the reasoning
block, the envelope it writes a call in, and — for the channel formats — the headers that
route a turn (`<|start|>assistant to=user<|message|>`). One `Parser` declares all three,
because they are one grammar: the header is what says whether the text after it is
reasoning, a call, or the answer.

The `Segmenter` holds back only what could still become a marker, which is how the first
half of `<tool_call>` stops reaching the client before anyone knows it was one: `<` alone
waits, `<tool` followed by `ing` leaves whole on the same push. The held text never exceeds
the longest marker minus one character — except inside a header, which is buffered whole
because nothing can be said about the text under it until the header closes.

Every channel but one is held the same way, and by the same rule: what could still be a
marker waits, and everything before it leaves. Reasoning streams because someone reads it
while it is written; the tool channel streams because a four-kilobyte argument held until
its envelope closes is a call the client cannot see until the generation is over. A header
is the exception — it routes, and nothing can be said about the text under one until it
closes, so it is buffered whole and then leaves on a channel of its own for the dialects to
drop.

Streaming the tool channel is not the same as deciding a call was made. This machine says
which channel text is on and never that it is a call: an envelope that turns out to spell
none goes back to the client as prose, and what does that is the dialect side (`Calls`),
which holds the fragments until a reader names something. The rule that survives is the one
that matters — nothing is announced as a call before it has a name.

Markers stay in the text of the segment they delimit: the segments of a generation
concatenate back into exactly what the model wrote.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeIs

__all__ = [
    "CallDelta",
    "Channel",
    "Headers",
    "MalformedToolCall",
    "Parser",
    "Segment",
    "Segmenter",
    "ToolCall",
    "ToolFamily",
    "ToolReader",
    "assemble",
]

type Channel = Literal["content", "reasoning", "tool", "header"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class CallDelta:
    """What a reader can say about one call now.

    `index` counts calls from the start of the generation and is what every dialect keys its
    frames by. `name` arrives exactly once per index, on the first delta of that call, and
    never after — a client that has been told a call is named cannot be told otherwise.

    `arguments` is the **JSON text of the arguments**, in fragments: concatenating every
    fragment of one index gives a JSON object, which is what `function.arguments`,
    `input_json_delta.partial_json` and `function_call_arguments.delta` all carry. It is not
    the source text the model wrote, and it cannot be: `<arg_value>3</arg_value>` and
    `days=3` are calls in two dialects whose source is not JSON at all, and a field that
    sometimes carries JSON and sometimes carries something else would have to be translated
    once per dialect on the way out.

    A fragment is not required to be valid JSON on its own — `{`, `"city": "Rio"` and `}` are
    three of them — only to concatenate into one. How finely a family can cut them is the
    family's: a dialect that learns a value's type only when the value ends emits one
    fragment per argument, and that is the floor, not a shortcoming.
    """

    index: int
    name: str | None
    arguments: str
    closed: bool


class ToolReader(Protocol):
    """One generation's reading of one family's envelopes.

    `push` takes the next piece of decoded text; `finish` takes nothing and reports what the
    end of the generation resolved. Both answer with deltas in the order they became sayable,
    and a reader holds only what it cannot decide yet.
    """

    def push(self, text: str) -> tuple[CallDelta, ...]: ...

    def finish(self) -> tuple[CallDelta, ...]: ...


class MalformedToolCall(ValueError):
    """An envelope the model wrote and did not fill with a call.

    Reported rather than skipped: a dropped envelope reaches the caller as a model that chose
    not to call anything, which is exactly the shape of a correct refusal. Only the
    whole-output path raises it — a stream has already handed out what it handed out, and
    what the model wrote badly is the client's to parse, which is what every OpenAI-shaped
    SDK already does.
    """

    def __init__(self, envelope: str, reason: str) -> None:
        self.envelope = envelope
        self.reason = reason
        super().__init__(f"{reason}: {envelope!r}")


@dataclass(frozen=True)
class ToolFamily:
    """The envelope one dialect writes a call in.

    `start` and `end` are what the *stream* holds on to — the `Segmenter` opens its tool
    channel on the first and closes it on the second, so they are the spelling a model
    generates. A reader may know more than that: replaying a history, a template can write
    the same call in another spelling, and only the reader is ever shown that text.

    `write` is the reader's inverse: one call, in the spelling this family generates. It
    exists because some templates declare tools and never read `tool_calls` back — SmolLM3
    and Falcon-H1 tell the model to write `<tool_call>` and have no branch for replaying one,
    so the assistant's own call vanishes from every history they render and the next turn
    answers a result with no visible question. What goes into the content there is what the
    model itself wrote, which is what a history is.

    Declared by every family rather than by the ones with a caller today: it is the same
    totality contract the kernel layer keeps, and it doubles as the test the readers did not
    have — write a call, read it back, and the two have to agree without a literal in the
    test to agree with.
    """

    start: str
    end: str
    reader: Callable[[], ToolReader]
    write: Callable[[ToolCall], str]
    grammar: Callable[[Sequence[Mapping[str, object]], Callable[[str], str]], str] | None = None
    """The envelope as a grammar over the tools offered, for a request that asks the model to
    call *something* — `tool_choice: "required"`, Anthropic's `{"type": "any"}`, a named
    function. `None` when this family cannot express one exactly, and that is not a gap to be
    filled by approximation: a grammar is a promise that the decode cannot produce anything
    else, so a family that spells its arguments in a syntax the compiler does not know
    (`<arg_value>3</arg_value>`, `days=3`) declares none and the request is refused with the
    checkpoint named. Forcing a call by any looser means answers with a call the model may
    never have made, which is the thing the refusal exists to prevent.

    The first argument is the declared tools, in the shape the dialects build (`declared`),
    because what the grammar has to pin is the name as well as the arguments: a forced call to
    a function nobody offered is not an answer to the request either.

    The second is how to spell a marker as a grammar term, which the family cannot know on its
    own: `<tool_call>` is a single added id in Qwen's vocabulary and a run of bytes in one
    that never added it, and a grammar asking for the bytes of an added id stalls on the first
    token (measured). `Vocabulary.literal` is what answers, so the same family compiles
    correctly against either kind of checkpoint.
    """

    def parse_tool_call(self, output: str) -> tuple[ToolCall, ...]:
        """Every call in a whole generation, in the order the model wrote them. The same
        reader the stream uses, driven over the text in one push."""
        machine = self.reader()
        return assemble((*machine.push(output), *machine.finish()))


@dataclass(frozen=True)
class Headers:
    """How a channel dialect routes a turn: a header between an opener and the body marker,
    classified into the channel it opens.

    `openers` are the markers that begin one (`<|start|>`, harmony's bare `<|channel|>`);
    `body` is what ends it (`<|message|>`); `tail` is what a generation prompt ends on when
    the model starts inside a header — harmony and atem both cut the prompt at
    `<|start|>assistant`, so the first generated text is already the header's.

    `classify` reads the whole header and answers with the channel it opens and that
    channel's closer, or `None` for a content header — the routing is the dialect's own
    vocabulary (`analysis`, `to=self`, `commentary to=`), and only the dialect can read it.
    """

    openers: tuple[str, ...]
    body: str
    tail: str
    classify: Callable[[str], tuple[Channel, str] | None]


@dataclass(frozen=True)
class Parser:
    """One dialect, whole: recognized off the checkpoint's chat template, never off a table
    keyed by `model_type` — the template renders the assistant's own turns when it replays a
    history, so the spellings it writes are the spellings the model emits, and a table goes
    stale one checkpoint later.

    `reasoning` is the opener-literal spellings (`<think>`); a headers dialect routes
    reasoning through `classify` instead and declares none here. The budget and the
    display-side strip read `REASONING` in the package root, which is assembled from the
    dialects' own pairs.
    """

    recognizes: Callable[[str], bool]
    reasoning: tuple[tuple[str, str], ...]
    tools: ToolFamily | None = None
    headers: Headers | None = None


@dataclass(frozen=True)
class Segment:
    channel: Channel
    text: str


def _resume(parser: "Parser", prompt: str) -> tuple[Channel, str]:
    """The channel a generation starts in, read off the prompt it continues.

    Qwen3.6's template writes `<think>\\n` into the generation prompt, so the first token
    is already inside the block and the opener never arrives — without this the whole
    reasoning of every turn is labelled content. Harmony and atem cut the prompt at
    `<|start|>assistant`, which puts the first token inside a header instead.

    The marker has to be what the prompt *ends* on, because that is the only position that
    means the template put the model inside the block. Searching the whole prompt reads a
    `<think>` a user typed as an open block and labels the entire answer reasoning; and a
    conversation replaying an earlier turn's reasoning carries closed blocks, which any rule
    that scans backwards then has to unpick.

    Only a reasoning block or a header is inherited. An envelope is something the model
    opens: taking one from the prompt would let a `<tool_call>` a user typed hold back the
    entire answer.
    """
    if parser.headers is not None and prompt.rstrip().endswith(parser.headers.tail):
        return "header", ""
    tail = prompt.rstrip()
    for marker, closer in parser.reasoning:
        if tail.endswith(marker):
            return "reasoning", closer
    return "content", ""


_RESTATEMENT_FLOOR = 32
"""How much copied prompt is a restatement rather than a quotation.

Under it the text leaves: a reasoning block opening with a phrase that also appears in the
prompt is a model referring to what it was asked, and thirty characters of agreement is
what any two texts about the same subject produce. Over it the model is transcribing.
"""


class _Restatement:
    """The prompt copied back at the top of the reasoning channel.

    Some checkpoints open their analysis by writing the request out again, verbatim, before
    the first line of their own. It is generated text and it costs its own forwards, but it
    is the reader's own words handed back, and every client that speaks to this daemon would
    otherwise strip it separately.

    Held rather than decided on: while everything written so far occurs in the prompt, it
    might still be a restatement, and the moment it does not it is settled for the rest of
    the generation. Nothing is delayed that is not a candidate — the hold is bounded by how
    much of the prompt the model copies, which is exactly what is being dropped — and text
    that turns out to be short enough to be a quotation leaves whole and in order.

    Matched against the whole prompt and not against the last message alone: which part of
    the prompt a checkpoint copies is not something this class can know, and a system block
    written back is as much a restatement as a question is.
    """

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt
        self._held = ""
        self._settled = prompt == ""

    def take(self, text: str) -> str:
        """What leaves now — everything, once the question is settled, and nothing while the
        copy might still be running."""
        if self._settled:
            return text
        candidate = self._held + text
        if candidate.strip() == "" or self._occurs(candidate):
            self._held = candidate
            return ""
        # The copy ended somewhere inside this piece: the longest prefix still in the prompt.
        cut = len(self._held)
        while cut < len(candidate) and self._occurs(candidate[: cut + 1]):
            cut += 1
        self._settled = True
        self._held = ""
        return candidate[cut:] if len(candidate[:cut].strip()) >= _RESTATEMENT_FLOOR else candidate

    def rest(self) -> str:
        """The end of the generation with the copy never having broken off: the whole
        reasoning was the prompt written back, and it goes — or it was too short to have
        been that, and it leaves."""
        held, self._held = self._held, ""
        self._settled = True
        return "" if len(held.strip()) >= _RESTATEMENT_FLOOR else held

    def _occurs(self, text: str) -> bool:
        return text.lstrip() in self._prompt


class Segmenter:
    """One generation's state: the channel the text is in, and what is still held.

    `parser` is this checkpoint's dialect, as `parser_for` read it off the chat template.
    A checkpoint nobody recognized still gets the fallback parser one level up: reasoning
    spellings are the model's own and are watched regardless, while an envelope nobody can
    parse is prose, and holding it would suppress text that was never a call.

    `prompt` is the text this generation continues, which is what says whether it starts
    inside a block instead of at its opener.
    """

    def __init__(self, parser: Parser, *, prompt: str = "") -> None:
        self._headers = parser.headers
        opens: list[tuple[str, Channel, str]] = [
            (marker, "reasoning", closer) for marker, closer in parser.reasoning
        ]
        if self._headers is not None:
            # The header owns the routing: a tool opener of its own would race the header's
            # markers at the same index and decide the same thing twice.
            opens.extend((opener, "header", "") for opener in self._headers.openers)
        elif parser.tools is not None:
            opens.append((parser.tools.start, "tool", parser.tools.end))
        self._opens: tuple[tuple[str, Channel, str], ...] = tuple(opens)
        channel, closer = _resume(parser, prompt)
        self._channel: Channel = channel
        self._closer = closer
        self._held = ""
        self._restatement = _Restatement(prompt)

    def push(self, text: str) -> tuple[Segment, ...]:
        held = self._held + text
        out: list[Segment] = []
        while True:
            if self._channel == "content":
                opening = self._opening(held)
                if opening is None:
                    break
                index, channel, closer = opening
                if index > 0:
                    out.append(Segment("content", held[:index]))
                held, self._channel, self._closer = held[index:], channel, closer
                continue
            if self._channel == "header":
                routed = self._route(held, out)
                if routed is None:
                    self._held = held
                    return self._filtered(out)
                held = routed
                continue
            closing = held.find(self._closer)
            if closing < 0:
                break
            cut = closing + len(self._closer)
            out.append(Segment(self._channel, held[:cut]))
            held, self._channel, self._closer = held[cut:], "content", ""
        keep = len(held) - self._ambiguous(held)
        self._held = held[keep:]
        if keep > 0:
            out.append(Segment(self._channel, held[:keep]))
        return self._filtered(out)

    def _filtered(self, out: list[Segment]) -> tuple[Segment, ...]:
        """The reasoning channel with the prompt's own words taken back out of the top of it,
        and every other channel untouched. A segment left empty by it does not leave: an
        empty delta is a frame that says a model wrote nothing."""
        kept: list[Segment] = []
        for segment in out:
            if segment.channel != "reasoning":
                kept.append(segment)
                continue
            text = self._restatement.take(segment.text)
            if text:
                kept.append(Segment("reasoning", text))
        return tuple(kept)

    def flush(self) -> tuple[Segment, ...]:
        """What is still held when the generation ends: a prefix that never resolved, an
        envelope the token budget cut in half, or a header the stop token landed inside. All
        leave — the envelope is what `parse_tool_call` reports as a call that never closed,
        and dropping any of them would show the model writing less than it did.

        The one exception is the restatement, which is the whole point of holding it: a
        generation that ended with the copy still running wrote nothing but the prompt back.
        """
        out: list[Segment] = []
        if self._held:
            held, self._held = self._held, ""
            # Through the filter first, because it came *after* whatever the restatement is
            # still holding: settling on it and then emitting this would invert the two.
            out.extend(self._filtered([Segment(self._channel, held)]))
        rest = self._restatement.rest()
        if rest:
            out.insert(0, Segment("reasoning", rest))
        return tuple(out)

    def _route(self, held: str, out: list[Segment]) -> str | None:
        """One header, once its body marker arrives: classified into the channel it opens.

        The header leaves on its own channel for reasoning and content — it routes, it is
        nobody's prose — but stays *in* a tool segment: harmony's reader finds the call's
        name in it (`to=functions.f`), and a reader handed the body alone reads an envelope
        with no call. `None` while the marker is still missing, which holds the header
        whole: nothing can be said about the text under one until it closes.
        """
        assert self._headers is not None
        at = held.find(self._headers.body)
        if at < 0:
            return None
        cut = at + len(self._headers.body)
        header, rest = held[:cut], held[cut:]
        routed = self._headers.classify(header)
        if routed is None:
            out.append(Segment("header", header))
            self._channel, self._closer = "content", ""
            return rest
        channel, closer = routed
        if channel == "tool":
            self._channel, self._closer = "tool", closer
            return held
        out.append(Segment("header", header))
        self._channel, self._closer = channel, closer
        return rest

    def _opening(self, held: str) -> tuple[int, Channel, str] | None:
        candidates: list[tuple[int, Channel, str]] = [
            (index, channel, closer)
            for marker, channel, closer in self._opens
            if (index := held.find(marker)) >= 0
        ]
        return min(candidates, default=None)

    def _ambiguous(self, held: str) -> int:
        """How much of the tail could still be the start of a marker. Every complete one was
        taken above, so what matches here is a proper prefix and never a whole marker."""
        markers = (
            tuple(marker for marker, _, _ in self._opens)
            if self._channel == "content"
            else (self._closer,)
        )
        longest = max((len(marker) for marker in markers), default=1) - 1
        for size in range(min(longest, len(held)), 0, -1):
            if any(marker.startswith(held[-size:]) for marker in markers):
                return size
        return 0


@dataclass
class _Building:
    name: str | None = None
    arguments: str = ""
    closed: bool = False


def _object(value: object) -> TypeIs[dict[str, object]]:
    """What `isinstance` cannot say on its own: a mapping `json.loads` built has string keys,
    because the format has no other kind. Nothing here reads deeper than one level, so the
    values stay `object`."""
    return isinstance(value, dict)


def _arguments(building: _Building) -> Mapping[str, object]:
    if not building.arguments:
        # A call with no arguments member at all, which the templates do write.
        return {}
    try:
        payload = json.loads(building.arguments)
    except json.JSONDecodeError as error:
        raise MalformedToolCall(
            building.arguments, f"the arguments did not parse as JSON ({error.msg})"
        ) from error
    if not _object(payload):
        raise MalformedToolCall(
            building.arguments, "the arguments did not parse as a JSON object"
        )
    return payload


def assemble(deltas: tuple[CallDelta, ...]) -> tuple[ToolCall, ...]:
    """The deltas of a whole generation, back into the calls they spell.

    Public because a streaming caller cannot use `parse_tool_call`: it has already driven a
    reader over the text to hand out fragments, and driving a second one over the same text
    is how two machines come to disagree about it. What it holds instead is the deltas, and
    this is what turns them back into calls.

    Every way an envelope can fail to be a call raises here and none of them is skipped. A
    call that never closed is one the generation cut in half — the marker missing, or the
    structure inside it left open, which is the same cut in the other spelling; one that
    closed without ever naming anything is an envelope the model filled with something that
    is not a call.
    """
    building: dict[int, _Building] = {}
    for delta in deltas:
        entry = building.setdefault(delta.index, _Building())
        if delta.name is not None:
            entry.name = delta.name
        entry.arguments += delta.arguments
        entry.closed = entry.closed or delta.closed
    calls: list[ToolCall] = []
    for entry in building.values():
        if not entry.closed:
            raise MalformedToolCall(entry.arguments, "the call never closes")
        if entry.name is None:
            raise MalformedToolCall(entry.arguments, "the call has no name")
        calls.append(ToolCall(entry.name, _arguments(entry)))
    return tuple(calls)
