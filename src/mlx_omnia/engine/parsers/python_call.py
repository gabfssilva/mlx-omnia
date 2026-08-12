"""A call written as Python source: LFM2.5's envelope.

    <|tool_call_start|>[get_weather(city='Rio', days=3)]<|tool_call_end|>

Named for the syntax and not for the checkpoint, like every other module here. What is
inside the markers is a Python **list of calls**, so one envelope spells several and
`CallDelta.index` moves inside it — the envelope is what the segmenter suppresses, and the
calls are what comes out of it.

Nothing here evaluates anything. `ast.literal_eval` runs over each argument's value alone,
which is the one call in the standard library that refuses a function call; the names and
the shape around the values are found by scanning. An argument whose value is not a literal
is a `MalformedToolCall` and the envelope goes back as content — the model wrote something
this cannot read, and reading it wrong would execute something nobody asked for.

The scan tracks bracket depth and string state because the separators are not separators
everywhere: the comma in `b="x, y"` is inside a string, and the one in `c=[1, 2]` is inside
a list. Splitting on `,` reads three arguments where the model wrote one.

Values are Python literals and the field is JSON, so `True` becomes `true` and `'Rio'`
becomes `"Rio"` — the translation is `json.dumps` over what `literal_eval` produced, and it
is why this family cannot stream a value before that value has ended.
"""

import ast
import json
from collections.abc import Iterator

from mlx_omnia.engine.parsers.envelope import Body, EnvelopeScanner
from mlx_omnia.engine.parsers.protocol import (
    CallDelta,
    MalformedToolCall,
    Parser,
    ToolCall,
    ToolFamily,
)
from mlx_omnia.engine.parsers.qwen import REASONING

__all__ = ["PARSER", "PythonCallReader"]

_START = "<|tool_call_start|>"
_END = "<|tool_call_end|>"

_OPEN = "([{"
_CLOSE = ")]}"


def _split(text: str, separator: str) -> list[str]:
    """`text` cut on `separator` where it is not inside a bracket or a string.

    The one piece of scanning this module cannot borrow: `str.split` is what reads
    `b="x, y"` as two arguments, and every family that spells arguments as source has the
    same problem in its own punctuation.
    """
    parts: list[str] = []
    depth = 0
    quote = ""
    escape = False
    start = 0
    for at, character in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == quote:
                quote = ""
            continue
        if character in "\"'":
            quote = character
        elif character in _OPEN:
            depth += 1
        elif character in _CLOSE:
            depth -= 1
        elif character == separator and depth == 0:
            parts.append(text[start:at])
            start = at + 1
    parts.append(text[start:])
    return [part for part in (piece.strip() for piece in parts) if part]


def _literal(text: str) -> str:
    """One argument's value as JSON text.

    `literal_eval` and not `eval`: it accepts literals and the containers of literals and
    refuses everything else, so `d=open("f")` raises here instead of running.
    """
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as error:
        raise MalformedToolCall(text, "the argument is not a literal") from error
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as error:
        # A literal Python can spell and JSON cannot: a set, a tuple key, complex.
        raise MalformedToolCall(text, "the argument has no JSON value") from error


def _arguments(text: str) -> Iterator[tuple[str, str]]:
    """The `name=value` pairs of one call, each value already JSON."""
    for argument in _split(text, ","):
        at = argument.find("=")
        if at < 0:
            raise MalformedToolCall(argument, "the argument is positional and has no name")
        yield argument[:at].strip(), _literal(argument[at + 1 :].strip())


class PythonCallReader:
    """The whole envelope at once.

    Unlike the element families there is nothing to hand out early: a Python list is only
    a list once its bracket closes, and an argument's type is only known at its last
    character. What this gains by waiting is the one thing this format makes hard — knowing
    where a call ends — and what it costs is fragments no finer than one call.
    """

    def __init__(self) -> None:
        self._scan = EnvelopeScanner(_START, _END)
        self._held = ""
        self._envelope = -1
        self._index = -1

    def push(self, text: str) -> tuple[CallDelta, ...]:
        return self._read(self._scan.push(text))

    def finish(self) -> tuple[CallDelta, ...]:
        out = self._read(self._scan.finish())
        if not self._scan.inside:
            return out
        return (*out, CallDelta(max(self._index + 1, 0), None, "", False))

    def _read(self, bodies: tuple[Body, ...]) -> tuple[CallDelta, ...]:
        out: list[CallDelta] = []
        for body in bodies:
            if body.index != self._envelope:
                self._envelope, self._held = body.index, ""
            self._held += body.text
            if not body.closed:
                continue
            out.extend(self._calls(self._held))
            self._held = ""
        return tuple(out)

    def _calls(self, body: str) -> Iterator[CallDelta]:
        listed = body.strip()
        # The brackets are the list's and are dropped; a body without them is one bare call,
        # which is what a model writes when it forgets them and is still a call.
        inner = listed[1:-1] if listed.startswith("[") and listed.endswith("]") else listed
        # By depth and not by the separator: the commas between calls and the commas between
        # a call's own arguments are the same character, and only depth tells them apart.
        calls = _split(inner, ",")
        if not calls:
            # An envelope holding no call. It still goes out, so the whole-output path sees
            # an envelope rather than a turn that called nothing.
            self._index += 1
            yield CallDelta(self._index, None, "", False)
            return
        for call in calls:
            self._index += 1
            at = call.find("(")
            if at < 0 or not call.endswith(")"):
                yield CallDelta(self._index, None, "", False)
                continue
            name = call[:at].strip()
            pairs = list(_arguments(call[at + 1 : -1]))
            body_text = ", ".join(f"{json.dumps(key)}: {value}" for key, value in pairs)
            yield CallDelta(self._index, name, f"{{{body_text}}}", True)


def write(call: ToolCall) -> str:
    """The call as Python source. `repr` and not `json.dumps`: the format is Python, so a
    string is quoted the way Python quotes one and `True` is not `true`."""
    arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    return f"{_START}[{call.name}({arguments})]{_END}"


PARSER = Parser(
    recognizes=lambda source: _START in source,
    reasoning=(REASONING,),
    tools=ToolFamily(start=_START, end=_END, reader=PythonCallReader, write=write),
)
