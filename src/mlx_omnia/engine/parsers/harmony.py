"""Harmony's envelope: a recipient in the header, `<|call|>` closing the body.

The channel alone is not an envelope — harmony writes the preamble to the user on commentary
too, closed by `<|end|>` and not by `<|call|>` — so what opens one is the recipient. That is
also what makes this family's reader see more than the stream does: generating, the model
writes the channel first and the recipient after it; replaying a history, the template writes
the same recipient into the role header instead. The family's `start` is the generated spelling,
because that is the only one the `Segmenter` is ever shown, while the reader opens on
`to=functions.` and reads both.

Builtin namespaces (`browser`, `python`) are never rendered into our prompts and there is
nothing on the other side to route them to, which the opener excludes by construction.
"""

import json
import re

from mlx_omnia.engine.parsers.envelope import Body, EnvelopeScanner
from mlx_omnia.engine.parsers.protocol import (
    CallDelta,
    Channel,
    Headers,
    Parser,
    ToolCall,
    ToolFamily,
)

__all__ = ["PARSER", "REASONING"]

REASONING = ("<|channel|>analysis", "<|end|>")

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


def _classify(header: str) -> tuple[Channel, str] | None:
    """The channel a harmony header opens. The recipient is what makes an envelope — the
    preamble to the user rides the commentary channel too, closed by `<|end|>` and not by
    `<|call|>` — and `analysis` is the reasoning channel gpt-oss opens every turn on.
    Anything else (`final`, bare commentary) is the answer."""
    if REASONING[0] in header:
        return "reasoning", REASONING[1]
    if _START in header:
        return "tool", _END
    return None


def write(call: ToolCall) -> str:
    """One call in the spelling the model generates — the recipient on the channel, which is
    the form the reader opens on. `<|constrain|>json` is what the format writes when the body
    is JSON, and the body always is here."""
    payload = json.dumps(call.arguments, ensure_ascii=False)
    return f"{_START}functions.{call.name} <|constrain|>json{_BODY}{payload}{_END}"


PARSER = Parser(
    recognizes=lambda source: _CHANNEL in source,
    reasoning=(),
    tools=ToolFamily(start=_START, end=_END, reader=HarmonyReader, write=write),
    headers=Headers(
        openers=("<|start|>", "<|channel|>"),
        body=_BODY,
        tail="<|start|>assistant",
        classify=_classify,
    ),
)
