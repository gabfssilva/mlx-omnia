"""`/api/anthropic/v1/*` — the dialect Claude Code speaks.

Three things separate it from OpenAI's. The system prompt is a field of the request and has
to leave as a turn of the conversation. The stream is named events, and a frame whose name is
missing is a frame the SDK never sees. The error envelope is its own, which is what a client's
own error mapping reads.

Tools make the block list earn itself: a call is a `tool_use` block of the assistant's message
and a result is a `tool_result` block of the *user's* — one message carries every result of a
round — while the conversation the engine takes has one turn per result. Reasoning is a
`thinking` block rather than text, and `stop_sequences` are honoured over the characters the
model wrote, because the engine's `stop` is a set of token ids.
"""

from mlx_omnia.server.api.anthropic.codec import encode_error
from mlx_omnia.server.api.anthropic.routes import router

__all__ = ["encode_error", "router"]
