"""Reading a tool call back out of the text the model generated.

A family is an envelope plus a parser for what it wraps: `<tool_call>{json}</tool_call>`
for Qwen, harmony's channels for GPT-OSS. Which one a checkpoint speaks is sniffed from
its chat template — the template renders the assistant's own calls when it replays a
history, so the markers it writes are the markers the model emits, and a table keyed by
`model_type` goes stale one checkpoint later.

Matching runs over decoded text, never over ids: the `>` closing a marker merges with the
next byte in tokenization, and an id-level match loses the call (a bug mlx-lm already
paid for). Whole envelopes only — holding a partial one while it is still ambiguous is
the stream's job, not this module's.

What this does not do is check the call against the tool's schema. An argument the schema
never declared goes through untouched: validating is the job of whoever defined the
function, and a call to a function nobody declared is a fact the caller has to see.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeIs

__all__ = [
    "HARMONY",
    "QWEN",
    "MalformedToolCall",
    "ToolCall",
    "ToolFamily",
    "tool_family",
]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, object]


class MalformedToolCall(ValueError):
    """An envelope the model wrote and did not fill with a call.

    Reported rather than skipped: a dropped envelope reaches the caller as a model that
    chose not to call anything, which is exactly the shape of a correct refusal.
    """

    def __init__(self, envelope: str, reason: str) -> None:
        self.envelope = envelope
        self.reason = reason
        super().__init__(f"{reason}: {envelope!r}")


@dataclass(frozen=True)
class ToolFamily:
    """The envelope one family writes a call in: the two markers that delimit it and the
    parser over the decoded output. The markers are what the stream holds on to; the
    parser only ever sees them closed."""

    start: str
    end: str
    parse_tool_call: Callable[[str], tuple[ToolCall, ...]]


def _json_object(value: object) -> TypeIs[dict[str, object]]:
    """What `isinstance` cannot say on its own: a mapping `json.loads` built has string
    keys, because the format has no other kind. Nothing here reads deeper than one
    level, so the values stay `object`."""
    return isinstance(value, dict)


def _object(value: object, envelope: str, what: str) -> dict[str, object]:
    if not _json_object(value):
        raise MalformedToolCall(envelope, f"{what} did not parse as a JSON object")
    return value


def _decode(text: str, what: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise MalformedToolCall(text, f"{what} did not parse as JSON ({error.msg})") from error
    return _object(payload, text, what)


_QWEN_START = "<tool_call>"
_QWEN_END = "</tool_call>"


def _parse_qwen(output: str) -> tuple[ToolCall, ...]:
    """Every `<tool_call>` in the output, in the order the model wrote them. Text outside
    the envelopes — the reasoning block, a sentence announcing the call — is not ours."""
    calls: list[ToolCall] = []
    rest = output
    while (opening := rest.find(_QWEN_START)) >= 0:
        rest = rest[opening + len(_QWEN_START) :]
        closing = rest.find(_QWEN_END)
        if closing < 0:
            raise MalformedToolCall(rest, "the envelope never closes")
        body = rest[:closing]
        payload = _decode(body, "the envelope")
        name = payload.get("name")
        if not isinstance(name, str):
            raise MalformedToolCall(body, "the call has no name")
        calls.append(ToolCall(name, _object(payload.get("arguments", {}), body, "the arguments")))
        rest = rest[closing + len(_QWEN_END) :]
    return tuple(calls)


_HARMONY_CHANNEL = "<|channel|>commentary"
# The channel alone is not an envelope: harmony writes the preamble to the user on it too,
# closed by `<|end|>` and not by `<|call|>`, so a stream holding on to the channel would
# hold back a whole answer and label it a call. What opens an envelope is the recipient in
# the header, in the spelling the model generates it — replaying a history the template
# writes the same recipient into the role header instead, which the parser reads and the
# stream cannot, since by then the message is already out.
_HARMONY_START = f"{_HARMONY_CHANNEL} to="
_HARMONY_END = "<|call|>"
_HARMONY_BOUNDARY = "<|start|>"
_HARMONY_BODY = "<|message|>"
_HARMONY_RECIPIENT = re.compile(r"to=functions\.([\w.-]+)")


def _parse_harmony(output: str) -> tuple[ToolCall, ...]:
    """Harmony messages are separated by `<|start|>`, and each header ends at
    `<|message|>`; the prompt already carries the first `<|start|>assistant`, so the
    generated text opens mid-header.

    What makes a message a call is the pair the protocol defines it by: a `functions.`
    recipient in the header and `<|call|>` closing the body — the token that *is* the
    invocation signal. The channel is not read: the system message the template writes
    sends calls to commentary, but a model that answers on another channel and still
    addresses a declared function has called it.

    The recipient is what opens the envelope, so a message that has one and no `<|call|>`
    is a call the generation cut in half — reported, like Qwen's unterminated one, rather
    than skipped into a silent "the model called nothing".
    """
    calls: list[ToolCall] = []
    for message in output.split(_HARMONY_BOUNDARY):
        header, marked, body = message.partition(_HARMONY_BODY)
        recipient = _HARMONY_RECIPIENT.search(header)
        if recipient is None:
            continue
        if not marked or _HARMONY_END not in body:
            raise MalformedToolCall(message, "the envelope never closes")
        arguments = body[: body.index(_HARMONY_END)]
        calls.append(ToolCall(recipient.group(1), _decode(arguments, "the arguments")))
    return tuple(calls)


QWEN = ToolFamily(_QWEN_START, _QWEN_END, _parse_qwen)
HARMONY = ToolFamily(_HARMONY_START, _HARMONY_END, _parse_harmony)

# Qwen3.6 keeps `<tool_call>` and changes everything inside it: the arguments are XML
# elements, not JSON. In the template source the newline is the two characters of a Jinja
# string literal, which is how every checkpoint in circulation spells it.
_QWEN_XML = "<tool_call>\\n<function="


def tool_family(template_source: str) -> ToolFamily | None:
    """Which envelope this checkpoint spells a call in, read off its chat template source.

    Unknown is an answer, not a failure: a family chosen by resemblance parses an envelope
    into a call that was never made. Qwen3.6's XML spelling lands here — it carries Qwen's
    marker and none of its contents.
    """
    if _HARMONY_CHANNEL in template_source:
        return HARMONY
    if _QWEN_XML in template_source:
        return None
    return QWEN if _QWEN_START in template_source else None
