"""`<tool_call>` around a JSON object with `name` and `arguments`.

The spelling every Qwen3 checkpoint writes, and with it Ling-3.0 and Laguna-S. Qwen3.6 keeps
the marker and fills it with XML instead, which is why the two recognizers are told apart by
what follows the marker in the template source and not by the marker alone.

Reading it as it arrives is the one family that needs a scanner of its own. The name is a
value *inside* the object, so nothing can be said about the call until the member carrying it
closes; the arguments are a value whose source text has to leave while the object is still
open. `json.loads` answers whole documents and can do neither.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from mlx_omnia.engine.parsers.envelope import Body, EnvelopeScanner
from mlx_omnia.engine.parsers.protocol import CallDelta, Parser, ToolCall, ToolFamily

__all__ = ["ARG_KEY_BODY", "END", "PARSER", "REASONING", "START", "XML_BODY"]

REASONING = ("<think>", "</think>")

START = "<tool_call>"
END = "</tool_call>"
"""The marker three families share. Public because they share it: a module that spelled it
again would be a second place deciding what an envelope is."""

XML_BODY = "<tool_call>\\n<function="
"""Qwen3.6's body. In the template source the newline is the two characters of a Jinja string
literal, which is how every checkpoint in circulation spells it."""

ARG_KEY_BODY = "<arg_key>"
"""Ling-3.0's and Laguna's body.

Both discriminators live here, in the module that owns the marker, rather than in the
families they name. The recognizers have to agree about which body belongs to whom, and a
string written in two modules is how two of them come to disagree — which is the bug this
constant exists to close: on the marker alone this dialect claimed those templates, its
scanner read their XML as malformed JSON, and every call they wrote reached the client as
prose with nothing raised anywhere.
"""

_NAME = "name"
_ARGUMENTS = "arguments"

_OBJECT, _KEY, _COLON, _VALUE = range(4)


@dataclass(frozen=True)
class _Member:
    key: str
    text: str
    closed: bool


class _Members:
    """Every member of one JSON object as it arrives: a key, the source text of its value in
    fragments, and whether that value has ended.

    Strings and escapes are tracked because a `}` inside a string closes nothing, and depth
    because the arguments are an object themselves. What this does not do is validate — a
    value that is not JSON streams out the way the model wrote it, and whoever needs it as a
    mapping parses it at the end.
    """

    def __init__(self) -> None:
        self._mode = _OBJECT
        self._key = ""
        self._depth = 0
        self._string = False
        self._escape = False
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the object itself closed. An envelope whose closing marker arrived with
        this still false is a call the generation cut in half."""
        return self._closed

    def push(self, text: str) -> tuple[_Member, ...]:
        out: list[_Member] = []
        held: list[str] = []

        def flush(closed: bool) -> None:
            if held or closed:
                out.append(_Member(self._key, "".join(held), closed))
                held.clear()

        for character in text:
            if self._closed:
                break
            if self._mode == _OBJECT:
                if character == "{":
                    self._mode = _KEY
            elif self._mode == _KEY:
                self._key_character(character)
            elif self._mode == _COLON:
                if character == ":":
                    self._mode, self._depth = _VALUE, 0
            elif self._value_character(character, held):
                flush(True)
        flush(False)
        return tuple(out)

    def _key_character(self, character: str) -> None:
        if self._string:
            if self._escape:
                self._escape = False
            elif character == "\\":
                self._escape = True
            elif character == '"':
                self._string, self._mode = False, _COLON
                return
            self._key += character
        elif character == '"':
            self._string, self._key = True, ""
        elif character == "}":
            self._closed = True

    def _value_character(self, character: str, held: list[str]) -> bool:
        """One character of a value, appended to `held`. True when the value just ended."""
        if self._string:
            held.append(character)
            if self._escape:
                self._escape = False
            elif character == "\\":
                self._escape = True
            elif character == '"':
                self._string = False
                if self._depth == 0:
                    self._mode = _KEY
                    return True
            return False
        if character == '"':
            self._string = True
            held.append(character)
            return False
        if character in "{[":
            self._depth += 1
            held.append(character)
            return False
        if character in "}]":
            if self._depth == 0:
                # The object's own closer, which is also what ends an undelimited scalar.
                self._closed = True
                return True
            self._depth -= 1
            held.append(character)
            if self._depth == 0:
                self._mode = _KEY
                return True
            return False
        if self._depth > 0:
            held.append(character)
            return False
        if character == ",":
            self._mode = _KEY
            return True
        if character.isspace():
            # Before the value it is not the value; after a scalar it is what ends it.
            if not held:
                return False
            self._mode = _KEY
            return True
        held.append(character)
        return False


def _string(text: str) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


class QwenReader:
    def __init__(self) -> None:
        self._scan = EnvelopeScanner(START, END)
        self._members = _Members()
        self._envelope = -1
        self._name = ""
        self._named = False
        self._pending = ""

    def push(self, text: str) -> tuple[CallDelta, ...]:
        return self._read(self._scan.push(text))

    def finish(self) -> tuple[CallDelta, ...]:
        out = self._read(self._scan.finish())
        if not self._scan.inside:
            return out
        # An envelope the generation stopped inside, which may have said nothing at all. It
        # leaves anyway: an opened envelope nobody hears about is one the whole-output path
        # cannot report, and a dropped envelope reads as a model that called nothing.
        return (*out, CallDelta(self._scan.index, None, "", False))

    def _read(self, bodies: tuple[Body, ...]) -> tuple[CallDelta, ...]:
        out: list[CallDelta] = []
        for body in bodies:
            if body.index != self._envelope:
                self._envelope = body.index
                self._members = _Members()
                self._name, self._named, self._pending = "", False, ""
            for member in self._members.push(body.text):
                out.extend(self._member(body.index, member))
            if body.closed:
                # `closed` is the *call* closing and not the marker: an envelope whose object
                # the generation cut in half is a call that never closed, and saying
                # otherwise would hand the whole-output path a call with no arguments instead
                # of an error. The delta goes out either way, so that an envelope that said
                # nothing is still an envelope somebody saw.
                out.append(CallDelta(body.index, None, "", self._members.closed))
        return tuple(out)

    def _member(self, index: int, member: _Member) -> tuple[CallDelta, ...]:
        if member.key == _NAME and not self._named:
            self._name += member.text
            if not member.closed or (name := _string(self._name)) is None:
                return ()
            self._named = True
            held, self._pending = self._pending, ""
            return (CallDelta(index, name, held, False),)
        if member.key == _ARGUMENTS:
            if self._named:
                return (CallDelta(index, None, member.text, False),)
            # The key order is the model's, and the name is the one thing a first delta
            # cannot go out without. Held until it resolves, bounded by the object.
            self._pending += member.text
        return ()


def _recognizes(source: str) -> bool:
    """`<tool_call>` and neither of the two bodies that are not this one.

    Three families share the marker and only the body tells them apart, so the marker alone
    is not a recognizer — it is how this dialect came to claim Ling-3.0 and Laguna, whose
    `<arg_key>` bodies its scanner reads as malformed JSON. A malformed call is dropped, and
    a dropped call reaches the client as prose: the model appears to have answered in text,
    with nothing raised anywhere.

    The two exclusions are imported from the families that own them rather than spelled here,
    because a marker written twice is how two recognizers come to disagree about one
    template.
    """
    return START in source and XML_BODY not in source and ARG_KEY_BODY not in source


def write(call: ToolCall) -> str:
    """One call as this family generates it. `separators` is jinja's own `tojson` default,
    which is what the templates that do replay a call write."""
    payload = {"name": call.name, "arguments": call.arguments}
    return f"{START}{json.dumps(payload, ensure_ascii=False)}{END}"


def grammar(
    tools: Sequence[Mapping[str, object]], literal: Callable[[str], str]
) -> str:
    """`<tool_call>` around an object whose `name` is one of the offered functions and whose
    `arguments` obey that function's own schema.

    One `anyOf` branch per tool and not a free-form object: the branch is what ties the name
    to the schema, so a grammar that allowed any offered name beside any offered arguments
    would guarantee a well-formed call to the wrong function.
    """
    branches = [
        {
            "type": "object",
            "properties": {
                "name": {"const": entry["name"]},
                "arguments": entry.get("parameters") or {"type": "object"},
            },
            "required": ["name", "arguments"],
        }
        for entry in (_function(tool) for tool in tools)
    ]
    body = json.dumps({"anyOf": branches})
    return f"start: {literal(START)} body {literal(END)}\nbody: %json {body}"


def _function(tool: Mapping[str, object]) -> Mapping[str, object]:
    """The dialects nest a function under `function`, which is the shape transformers
    documents and the templates read."""
    nested = tool.get("function")
    return nested if isinstance(nested, Mapping) else tool


PARSER = Parser(
    recognizes=_recognizes,
    reasoning=(REASONING,),
    tools=ToolFamily(start=START, end=END, reader=QwenReader, write=write, grammar=grammar),
)
