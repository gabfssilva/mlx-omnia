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
from dataclasses import dataclass

from sideros.tools.envelope import Body, EnvelopeScanner
from sideros.tools.protocol import CallDelta, ToolFamily

__all__ = ["FAMILY"]

_START = "<tool_call>"
_END = "</tool_call>"
# Qwen3.6 keeps `<tool_call>` and changes everything inside it. In the template source the
# newline is the two characters of a Jinja string literal, which is how every checkpoint in
# circulation spells it.
_XML = "<tool_call>\\n<function="

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
        self._scan = EnvelopeScanner(_START, _END)
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


FAMILY = ToolFamily(
    start=_START,
    end=_END,
    recognizes=lambda source: _START in source and _XML not in source,
    reader=QwenReader,
)
