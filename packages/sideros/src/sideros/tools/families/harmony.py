"""Harmony's envelope: a recipient in the header, `<|call|>` closing the body.

The channel alone is not an envelope — harmony writes the preamble to the user on commentary
too, closed by `<|end|>` and not by `<|call|>` — so what opens one is the recipient. That is
also what makes this family's reader see more than the stream does: generating, the model
writes the channel first and the recipient after it; replaying a history, the template writes
the same recipient into the role header instead. `FAMILY.start` is the generated spelling,
because that is the only one the `Segmenter` is ever shown, while the reader opens on
`to=functions.` and reads both.

Builtin namespaces (`browser`, `python`) are never rendered into our prompts and there is
nothing on the other side to route them to, which the opener excludes by construction.
"""

import re

from sideros.tools.envelope import Body, EnvelopeScanner
from sideros.tools.protocol import CallDelta, ToolFamily

__all__ = ["FAMILY"]

_CHANNEL = "<|channel|>commentary"
_START = f"{_CHANNEL} to="
_END = "<|call|>"
_RECIPIENT = "to=functions."
_BODY = "<|message|>"
_NAME = re.compile(r"[\w.-]+")


class HarmonyReader:
    """The header is held until `<|message|>`, which is what carries the name; from there the
    body is the arguments and leaves as it arrives."""

    def __init__(self) -> None:
        self._scan = EnvelopeScanner(_RECIPIENT, _END)
        self._envelope = -1
        self._held = ""
        self._named = False

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
                self._envelope, self._held, self._named = body.index, "", False
            if self._named:
                if body.text:
                    out.append(CallDelta(body.index, None, body.text, False))
            else:
                self._held += body.text
                head, marked, rest = self._held.partition(_BODY)
                if marked and (name := _NAME.match(head)) is not None:
                    self._named, self._held = True, ""
                    out.append(CallDelta(body.index, name.group(), rest, False))
            if body.closed and self._named:
                out.append(CallDelta(body.index, None, "", True))
        return tuple(out)


FAMILY = ToolFamily(
    start=_START,
    end=_END,
    recognizes=lambda source: _CHANNEL in source,
    reader=HarmonyReader,
)
