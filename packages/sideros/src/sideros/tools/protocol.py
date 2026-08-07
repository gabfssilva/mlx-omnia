"""What a family is, and what reading one costs whoever is not calling a tool.

A family is an envelope plus a reader for what it wraps. The reader is **incremental**: it is
handed decoded text in whatever pieces the detokenizer produced and answers with what it can
say now, which is what lets the name of a call and its arguments leave while the envelope is
still open. There is no second, whole-output parser — `parse_tool_call` drives this same
reader over the whole text, because two machines over one generation do not agree.

Matching runs over decoded text, never over ids: the `>` closing a marker merges with the
next byte in tokenization, and an id-level match loses the call (a bug other engines already paid
for).

What this does not do is check a call against the tool's schema. An argument the schema never
declared goes through untouched: validating is the job of whoever defined the function, and a
call to a function nobody declared is a fact the caller has to see.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeIs

__all__ = [
    "CallDelta",
    "MalformedToolCall",
    "ToolCall",
    "ToolFamily",
    "ToolReader",
]


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
    `arguments` is the **source text** the model wrote, in fragments: concatenating them
    gives back exactly what it wrote, which is what the field carries on the wire.
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
    """The envelope one family writes a call in.

    `start` and `end` are what the *stream* holds on to — the `Segmenter` opens its tool
    channel on the first and closes it on the second, so they are the spelling a model
    generates. A reader may know more than that: replaying a history, a template can write
    the same call in another spelling, and only the reader is ever shown that text.

    `recognizes` reads the checkpoint's own chat template source. It lives with the family
    because that is where the knowledge is; the registry that calls it knows no marker.
    """

    start: str
    end: str
    recognizes: Callable[[str], bool]
    reader: Callable[[], ToolReader]

    def parse_tool_call(self, output: str) -> tuple[ToolCall, ...]:
        """Every call in a whole generation, in the order the model wrote them. The same
        reader the stream uses, driven over the text in one push."""
        machine = self.reader()
        return _assemble((*machine.push(output), *machine.finish()))


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


def _assemble(deltas: tuple[CallDelta, ...]) -> tuple[ToolCall, ...]:
    """The deltas of a whole generation, back into the calls they spell.

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
