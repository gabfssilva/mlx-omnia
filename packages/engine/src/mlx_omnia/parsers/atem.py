"""Muse-Glimmer's dialect: harmony's channels, Anthropic's envelope.

The turn structure is harmony's — `<|start|>assistant to=<recipient><|message|>`, `<|eom|>`
switching channels, `to=self` where gpt-oss says `analysis` — but the call is spelled as an
XML-ish block instead of JSON on a channel:

    <atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Paris</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

One envelope can carry several `invoke`s, so the reader counts calls by invoke and not by
envelope. The template writes strings raw and everything else through `tojson`, the same
convention as Qwen3.6's XML — read back the same way, except nothing is stripped: the
format's own instructions say spaces in string values are preserved.
"""

import json
from collections.abc import Iterator

from mlx_omnia.parsers.envelope import Body, EnvelopeScanner
from mlx_omnia.parsers.protocol import CallDelta, Channel, Headers, Parser, ToolFamily

__all__ = ["PARSER", "REASONING"]

REASONING = ("to=self", "<|eom|>")

_START = "<atem:function_calls>"
_END = "</atem:function_calls>"
_INVOKE = '<atem:invoke name="'
_INVOKE_END = "</atem:invoke>"
_PARAMETER = '<atem:parameter name="'
_PARAMETER_END = "</atem:parameter>"
_ATTRIBUTE_END = '">'


def _encoded(value: str) -> str:
    """The JSON text of a value the format left untyped. What parses is what the template's
    own `tojson` wrote; what does not is the string the model typed."""
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return json.dumps(value)
    return value


class AtemReader:
    def __init__(self) -> None:
        self._scan = EnvelopeScanner(_START, _END)
        self._envelope = -1
        self._held = ""
        self._call = -1
        self._inside = False
        self._key: str | None = None
        self._written = 0
        self._opened = 0

    def push(self, text: str) -> tuple[CallDelta, ...]:
        return self._read(self._scan.push(text))

    def finish(self) -> tuple[CallDelta, ...]:
        out = self._read(self._scan.finish())
        if not self._scan.inside:
            return out
        # An envelope the generation stopped inside. The invoke being read leaves as a call
        # that never closed; an envelope that named nothing yet still leaves, so that the
        # whole-output path sees it rather than a model that called nothing.
        index = self._call if self._inside else self._call + 1
        return (*out, CallDelta(index, None, "", False))

    def _read(self, bodies: tuple[Body, ...]) -> tuple[CallDelta, ...]:
        out: list[CallDelta] = []
        for body in bodies:
            if body.index != self._envelope:
                self._envelope, self._held = body.index, ""
                self._inside, self._key, self._written, self._opened = False, None, 0, 0
            self._held += body.text
            out.extend(self._drain())
            if body.closed:
                if self._inside:
                    # The envelope closed mid-invoke: a call cut in half, never closed.
                    out.append(CallDelta(self._call, None, "", False))
                    self._inside = False
                elif not self._opened:
                    # Closed without ever invoking: an envelope somebody has to see, and
                    # what it holds is a call with no name.
                    out.append(CallDelta(self._call + 1, None, "", True))
                self._held = ""
        return tuple(out)

    def _drain(self) -> Iterator[CallDelta]:
        while True:
            if not self._inside:
                name = self._between(_INVOKE)
                if name is None:
                    return
                self._call += 1
                self._opened += 1
                self._inside, self._written = True, 0
                yield CallDelta(self._call, name, "{", False)
            elif self._key is None:
                closing = self._held.find(_INVOKE_END)
                opening = self._held.find(_PARAMETER)
                if closing >= 0 and (opening < 0 or closing < opening):
                    self._held = self._held[closing + len(_INVOKE_END) :]
                    self._inside = False
                    yield CallDelta(self._call, None, "}", True)
                    continue
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
                    self._call, None, f"{separator}{json.dumps(key)}: {_encoded(value)}", False
                )

    def _between(self, opener: str) -> str | None:
        """The attribute between `opener` and the `\">` closing it, consumed. None while
        either end is still missing — what is held is one element's head."""
        at = self._held.find(opener)
        if at < 0:
            return None
        rest = self._held[at + len(opener) :]
        end = rest.find(_ATTRIBUTE_END)
        if end < 0:
            return None
        self._held = rest[end + len(_ATTRIBUTE_END) :]
        return rest[:end]

    def _until(self, closer: str) -> str | None:
        at = self._held.find(closer)
        if at < 0:
            return None
        value, self._held = self._held[:at], self._held[at + len(closer) :]
        return value


def _classify(header: str) -> tuple[Channel, str] | None:
    """The channel an atem header opens: `to=self` is the reasoning channel, `to=user` the
    answer, any other recipient a tool. A header with no recipient at all leaves the choice
    to the model's next text, which is the answer."""
    if REASONING[0] in header:
        return "reasoning", REASONING[1]
    if "to=user" in header:
        return None
    if "to=" in header:
        return "tool", _END
    return None


PARSER = Parser(
    recognizes=lambda source: _START in source,
    reasoning=(),
    tools=ToolFamily(start=_START, end=_END, reader=AtemReader),
    headers=Headers(
        # `<|eom|>` opens the next header when a turn continues past a tool call, which is
        # what keeps the separator out of the content channel.
        openers=("<|start|>", "<|eom|>"),
        body="<|message|>",
        tail="<|start|>assistant",
        classify=_classify,
    ),
)
