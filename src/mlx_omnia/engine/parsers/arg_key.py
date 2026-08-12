"""`<tool_call>` with the name on the marker's own line and the arguments as element pairs.

    <tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Rio</arg_value>
    <arg_key>days</arg_key>
    <arg_value>3</arg_value>
    </tool_call>

Ling-3.0 and Laguna write it, and the whitespace between the elements is not shared between
them — Laguna runs them together, Ling puts a newline after each `</arg_key>`. Nothing here
depends on that, which is the point of scanning for the elements rather than splitting on
lines.

It shares `<tool_call>` with two other families and is told apart by what the template puts
inside: this one by `<arg_key>`, Qwen3.6's by `<function=`, and Qwen's by neither. Sharing a
marker is exactly how a dialect comes to read another's envelope — before this module the
recognizer for Qwen claimed these templates on the marker alone, and every call they wrote
reached the client as text.

The arguments arrive untyped, so a value's JSON is only decidable once it has ended: this
family streams one fragment per argument, the floor `CallDelta` describes.
"""

import json
from collections.abc import Iterator

from mlx_omnia.engine.parsers.envelope import Body, EnvelopeScanner, encoded
from mlx_omnia.engine.parsers.protocol import CallDelta, Parser, ToolCall, ToolFamily
from mlx_omnia.engine.parsers.qwen import ARG_KEY_BODY, END, REASONING, START

__all__ = ["PARSER", "ArgKeyReader"]

_ARG_KEY_END = "</arg_key>"
_ARG_VALUE = "<arg_value>"
_ARG_VALUE_END = "</arg_value>"


class ArgKeyReader:
    def __init__(self) -> None:
        self._scan = EnvelopeScanner(START, END)
        self._envelope = -1
        self._held = ""
        self._named = False
        self._key: str | None = None
        self._written = 0

    def push(self, text: str) -> tuple[CallDelta, ...]:
        return self._read(self._scan.push(text))

    def finish(self) -> tuple[CallDelta, ...]:
        out = self._read(self._scan.finish())
        if not self._scan.inside:
            return out
        return (*out, CallDelta(self._scan.index, None, "", False))

    def _read(self, bodies: tuple[Body, ...]) -> tuple[CallDelta, ...]:
        out: list[CallDelta] = []
        for body in bodies:
            if body.index != self._envelope:
                self._envelope, self._held = body.index, ""
                self._named, self._key, self._written = False, None, 0
            self._held += body.text
            out.extend(self._drain(body.index, body.closed))
            if body.closed:
                out.append(CallDelta(body.index, None, "}" if self._named else "", self._named))
                self._held = ""
        return tuple(out)

    def _drain(self, index: int, closing: bool) -> Iterator[CallDelta]:
        while True:
            if not self._named:
                # The name runs from the marker to the first element, and only the envelope
                # closing proves there is no element coming: a call with no arguments at all
                # is `<tool_call>ping</tool_call>`, and its name is the whole body.
                at = self._held.find(ARG_KEY_BODY)
                if at < 0 and not closing:
                    return
                name = (self._held if at < 0 else self._held[:at]).strip()
                if not name:
                    return
                self._held = "" if at < 0 else self._held[at:]
                self._named = True
                yield CallDelta(index, name, "{", False)
            elif self._key is None:
                key = self._between(ARG_KEY_BODY, _ARG_KEY_END)
                if key is None:
                    return
                self._key = key.strip()
            else:
                value = self._between(_ARG_VALUE, _ARG_VALUE_END)
                if value is None:
                    return
                key, self._key = self._key, None
                separator = ", " if self._written else ""
                self._written += 1
                yield CallDelta(
                    index, None, f"{separator}{json.dumps(key)}: {encoded(value)}", False
                )

    def _between(self, opener: str, closer: str) -> str | None:
        """The text between the two markers, consumed. None while either is still missing —
        what is held is one element that has not finished arriving."""
        at = self._held.find(opener)
        if at < 0:
            return None
        rest = self._held[at + len(opener) :]
        end = rest.find(closer)
        if end < 0:
            return None
        self._held = rest[end + len(closer) :]
        return rest[:end]


def write(call: ToolCall) -> str:
    """Ling's spelling of the pairs. Laguna runs them together and reads either, because the
    reader scans for the elements instead of splitting on lines."""
    pairs = "".join(
        f"\n{ARG_KEY_BODY}{key}{_ARG_KEY_END}\n{_ARG_VALUE}{_written(value)}{_ARG_VALUE_END}"
        for key, value in call.arguments.items()
    )
    return f"{START}{call.name}{pairs}\n{END}"


def _written(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


PARSER = Parser(
    recognizes=lambda source: START in source and ARG_KEY_BODY in source,
    reasoning=(REASONING,),
    tools=ToolFamily(start=START, end=END, reader=ArgKeyReader, write=write),
)
