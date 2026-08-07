"""Qwen3.6's envelope: Qwen's marker, XML elements inside it.

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

The arguments are assembled into the JSON text the dialects carry, one member per element,
which is why this family streams **per parameter** rather than per character: the format
gives no type, so whether a value is a JSON literal or a string is only decidable once it has
ended. The template writes strings raw and everything else through `tojson`, and that is the
convention read back here.
"""

import json
from collections.abc import Iterator

from sideros.tools.envelope import Body, EnvelopeScanner
from sideros.tools.protocol import CallDelta, ToolFamily

__all__ = ["FAMILY"]

_START = "<tool_call>"
_END = "</tool_call>"
_XML = "<tool_call>\\n<function="

_FUNCTION = "<function="
_PARAMETER = "<parameter="
_PARAMETER_END = "</parameter>"
_CLOSE = ">"


def _encoded(value: str) -> str:
    """The JSON text of a value the format left untyped. What parses is what the template's
    own `tojson` wrote; what does not is the string the model typed."""
    body = value.strip("\n")
    try:
        json.loads(body)
    except json.JSONDecodeError:
        return json.dumps(body)
    return body


class QwenXmlReader:
    def __init__(self) -> None:
        self._scan = EnvelopeScanner(_START, _END)
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
            out.extend(self._drain(body.index))
            if body.closed:
                # An envelope that never named a function closes without closing a call: the
                # delta still goes out, so that the whole-output path sees the envelope.
                out.append(CallDelta(body.index, None, "}" if self._named else "", self._named))
                self._held = ""
        return tuple(out)

    def _drain(self, index: int) -> Iterator[CallDelta]:
        while True:
            if not self._named:
                name = self._between(_FUNCTION)
                if name is None:
                    return
                self._named = True
                yield CallDelta(index, name, "{", False)
            elif self._key is None:
                key = self._between(_PARAMETER)
                if key is None:
                    return
                self._key = key
            else:
                value = self._until(_PARAMETER_END)
                if value is None:
                    return
                key, self._key = self._key, None
                separator = ", " if self._written else ""
                self._written += 1
                yield CallDelta(
                    index, None, f"{separator}{json.dumps(key)}: {_encoded(value)}", False
                )

    def _between(self, opener: str) -> str | None:
        """The text between `opener` and the `>` closing it, consumed. None while either end
        is still missing — what is held is one element's head."""
        at = self._held.find(opener)
        if at < 0:
            return None
        rest = self._held[at + len(opener) :]
        end = rest.find(_CLOSE)
        if end < 0:
            return None
        self._held = rest[end + len(_CLOSE) :]
        return rest[:end]

    def _until(self, closer: str) -> str | None:
        at = self._held.find(closer)
        if at < 0:
            return None
        value, self._held = self._held[:at], self._held[at + len(closer) :]
        return value


FAMILY = ToolFamily(
    start=_START,
    end=_END,
    recognizes=lambda source: _XML in source,
    reader=QwenXmlReader,
)
