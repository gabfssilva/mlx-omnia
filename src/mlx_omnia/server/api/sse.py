"""SSE framing, byte for byte as each dialect writes it.

Three shapes are in use: named frames (`event:` plus `data:`) for Anthropic and
`/v1/responses`, bare `data:` frames closed by `[DONE]` for `chat/completions`, and bare
`data:` frames with no terminator for Gemini.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

MEDIA_TYPE = "text/event-stream"
DONE = "data: [DONE]\n\n"
KEEP_ALIVE = ": keep-alive\n\n"
"""A comment frame, which holds the connection open through a long prefill without reaching
any SDK's accumulator."""


def data(payload: Mapping[str, object]) -> str:
    """A bare frame: `chat/completions` and Gemini."""
    return f"data: {json.dumps(payload)}\n\n"


def named(event: str, payload: Mapping[str, object]) -> str:
    """A frame carrying its own event name: Anthropic and `/v1/responses`."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def sequenced(name: str, sequence: int, fields: Mapping[str, object]) -> str:
    """`/v1/responses`: the event name goes on the line and into the payload, beside the
    counter the SDK reads."""
    payload: dict[str, object] = {"type": name, "sequence_number": sequence, **fields}
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def ping() -> str:
    """Anthropic's own keep-alive, which is a real event rather than a comment."""
    return named("ping", {"type": "ping"})
