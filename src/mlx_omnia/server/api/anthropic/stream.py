"""The generation's events as this dialect's named frames."""

from collections.abc import AsyncIterator, Iterator, Mapping
from itertools import count

from mlx_omnia.engine.generate import Meter
from mlx_omnia.server.api import sse
from mlx_omnia.server.api.anthropic.encode import (
    STOP_REASONS,
    message_object,
    opening_usage,
    thought_block,
    use_block,
)
from mlx_omnia.server.generation.consume import Beat, ClosedToolCallDelta
from mlx_omnia.server.runtime.events import (
    Failed,
    Finished,
    ReasoningDelta,
    Started,
    TextDelta,
    ToolCallDelta,
    ToolCalls,
)


async def encode_stream(
    events: AsyncIterator[Beat],
    message_id: str,
    model: str,
    shown: bool,
    meter: Meter,
) -> AsyncIterator[str]:
    """The generation's events as this dialect's named frames.

    Blocks are opened one at a time and closed when the channel turns: reasoning and the answer
    are two content blocks of one message, in the order the model wrote them, and a delta whose
    type does not match the block it lands in is an accumulator that raises.
    """
    blocks = count()
    index: int | None = None
    kind: str | None = None

    def closed() -> Iterator[str]:
        nonlocal index, kind
        if index is not None:
            yield sse.named("content_block_stop", {"type": "content_block_stop", "index": index})
        index, kind = None, None

    def opened(wanted: str, block: Mapping[str, object]) -> Iterator[str]:
        """The block a channel writes into, opened on the first piece of it and not before: a
        generation that only called something has no text block."""
        nonlocal index, kind
        if kind == wanted:
            return
        yield from closed()
        index, kind = next(blocks), wanted
        yield sse.named(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": block},
        )

    def said(content: str) -> Iterator[str]:
        yield from opened("text", {"type": "text", "text": ""})
        assert index is not None
        yield sse.named(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": content},
            },
        )

    def reasoned(content: str) -> Iterator[str]:
        """The block goes out whether or not its text does: `display: "omitted"` is a client
        that does not want to read the reasoning, not one that wants to be told there was
        none."""
        yield from opened("thinking", thought_block(""))
        assert index is not None
        if shown and content:
            yield sse.named(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": content},
                },
            )

    def calling(delta: ToolCallDelta, close: bool) -> Iterator[str]:
        """A call's block, opened when the reader first names it and filled as it resolves.

        The arguments arrive as `input_json_delta`, which is the one shape the SDK's accumulator
        reads. A block is closed by the delta that says the call closed, so the next call's
        block can open while this one is already executable on the client.
        """
        if delta.name is not None and delta.id is not None:
            yield from opened(f"tool:{delta.index}", use_block(f"toolu_{delta.id}", delta.name, {}))
        assert index is not None
        if delta.arguments:
            yield sse.named(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": delta.arguments},
                },
            )
        if close:
            yield from closed()

    async for beat in events:
        match beat:
            case None:
                yield sse.ping()
            case Started():
                # `message_start` waits for the first piece because it carries `input_tokens`,
                # and the prompt is only counted once the template has rendered it on the other
                # side of the queue. The ping above is what holds the connection open until
                # then.
                yield sse.named(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": message_object(
                            message_id, model, [], None, opening_usage(meter)
                        ),
                    },
                )
            case ReasoningDelta(content):
                for frame in reasoned(content):
                    yield frame
            case TextDelta(content):
                for frame in said(content):
                    yield frame
            case ClosedToolCallDelta() as delta:
                for frame in calling(delta, close=True):
                    yield frame
            case ToolCallDelta() as delta:
                for frame in calling(delta, close=False):
                    yield frame
            case ToolCalls():
                pass
            case Failed(message):
                # The status is long gone — the response opened 200 the moment the first frame
                # went out — so the dialect's own event is the only place left to say it.
                yield sse.named(
                    "error", {"type": "error", "error": {"type": "api_error", "message": message}}
                )
                return
            case Finished(reason, usage, stop_sequence, _):
                for frame in closed():
                    yield frame
                yield sse.named(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": STOP_REASONS[reason],
                            "stop_sequence": stop_sequence,
                        },
                        "usage": {"output_tokens": usage.completion_tokens},
                    },
                )
                yield sse.named("message_stop", {"type": "message_stop"})
            case _:
                pass
