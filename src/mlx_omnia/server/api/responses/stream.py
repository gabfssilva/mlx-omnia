from collections.abc import AsyncIterator, Iterator
from itertools import count

from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api import sse
from mlx_omnia.server.api.responses.frames import call_item, message, part, response
from mlx_omnia.server.api.responses.models import ResponsesRequest
from mlx_omnia.server.api.responses.wire import Checked, document, failed
from mlx_omnia.server.generation.consume import ClosedToolCallDelta, Options, consume
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.runtime.events import (
    Failed,
    Finished,
    TextDelta,
    ToolCallDelta,
    ToolCalls,
    Usage,
)


def dialect(request: ResponsesRequest, tools: bool) -> Options:
    """How this route reads a generation: the reasoning block stays in the answer as the model
    wrote it, the turn ends on the budget alone — a call and a full budget is `incomplete`
    here, as it was — and there are no client stop sequences to hold for."""
    return Options(
        tools=tools,
        max_tokens=request.max_output_tokens,
        context_limit=None,
        reasoning="text",
        call_ends_turn=False,
    )


async def frames(
    job: Job,
    request: ResponsesRequest,
    *,
    request_id: str,
    message_id: str,
    created: int,
    checked: Checked | None,
    options: Options,
) -> AsyncIterator[str]:
    sequence = count()
    pieces: list[str] = []
    started = False
    output: list[dict[str, object]] = []
    items: dict[int, tuple[str, str, str]] = {}
    """Per call index: the item id, the call id and the name. Both ids are drawn from the one
    the generation minted when the reader first named the call, because they are what the
    client matches its `function_call_output` to."""

    def deltas(content: str) -> Iterator[str]:
        """The item and its content part are announced on the first text there is, not before:
        a generation that only called something has no message item at all."""
        nonlocal started
        if not content:
            return
        if not started:
            started = True
            yield sse.sequenced(
                "response.output_item.added",
                next(sequence),
                {"output_index": 0, "item": message(message_id, None)},
            )
            yield sse.sequenced(
                "response.content_part.added",
                next(sequence),
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": message_id,
                    "part": part(""),
                },
            )
        pieces.append(content)
        yield sse.sequenced(
            "response.output_text.delta",
            next(sequence),
            {
                "output_index": 0,
                "content_index": 0,
                "item_id": message_id,
                "delta": content,
                "logprobs": [],
            },
        )

    def message_done() -> Iterator[str]:
        """The text item closed, so that a call item can be opened after it.

        The dialect numbers items in the order they are announced, and the SDK's accumulator
        builds the response from that order. Idempotent: `started` says whether there is an
        item at all, and `output` whether it has already been closed.
        """
        if not started or output:
            return
        text = "".join(pieces)
        for event, payload in [
            ("response.output_text.done", {"text": text, "logprobs": []}),
            ("response.content_part.done", {"part": part(text)}),
        ]:
            yield sse.sequenced(
                event,
                next(sequence),
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": message_id,
                    **payload,
                },
            )
        yield sse.sequenced(
            "response.output_item.done",
            next(sequence),
            {"output_index": 0, "item": message(message_id, text)},
        )
        output.append(message(message_id, text))

    def calling(delta: ToolCallDelta) -> Iterator[str]:
        """One call item per index, opened when the reader names it and filled as it resolves.

        `arguments` is empty on `output_item.added` and arrives as
        `function_call_arguments.delta`: the SDK's accumulator concatenates those into the item
        it was told about.
        """
        if delta.name is not None and delta.id is not None:
            yield from message_done()
            items[delta.index] = (f"fc_{delta.id}", f"call_{delta.id}", delta.name)
            item_id, call_id, name = items[delta.index]
            yield sse.sequenced(
                "response.output_item.added",
                next(sequence),
                {
                    "output_index": len(output),
                    "item": call_item(item_id, call_id, name, None),
                },
            )
            output.append(call_item(item_id, call_id, name, ""))
        item_id, call_id, name = items[delta.index]
        at = next(i for i, entry in enumerate(output) if entry.get("id") == item_id)
        if delta.arguments:
            written = output[at]
            written["arguments"] = f"{written.get('arguments', '')}{delta.arguments}"
            yield sse.sequenced(
                "response.function_call_arguments.delta",
                next(sequence),
                {"output_index": at, "item_id": item_id, "delta": delta.arguments},
            )
        if isinstance(delta, ClosedToolCallDelta):
            arguments = str(output[at].get("arguments", ""))
            yield sse.sequenced(
                "response.function_call_arguments.done",
                next(sequence),
                {
                    "output_index": at,
                    "item_id": item_id,
                    "name": name,
                    "arguments": arguments,
                },
            )
            output[at] = call_item(item_id, call_id, name, arguments)
            yield sse.sequenced(
                "response.output_item.done",
                next(sequence),
                {"output_index": at, "item": output[at]},
            )

    def broke(usage: Usage, reason: str) -> str:
        """A generation that will not be completed, with the text that did arrive still in the
        item: what failed is the rest of it, and a client that already rendered those deltas is
        not told they never happened. `response.completed` is the frame the SDK accumulates
        into a final response, so a failure must not wear it."""
        return sse.sequenced(
            "response.failed",
            next(sequence),
            {
                "response": response(
                    request_id,
                    created,
                    request,
                    [message(message_id, "".join(pieces))] if started else [],
                    "failed",
                    usage,
                    error=reason,
                )
            },
        )

    made = False
    opened = response(request_id, created, request, [], "in_progress", None)
    yield sse.sequenced("response.created", next(sequence), {"response": opened})
    async for beat in consume(job, options):
        match beat:
            case None:
                yield sse.KEEP_ALIVE
            case TextDelta(text):
                for frame in deltas(text):
                    yield frame
            case ToolCallDelta():
                for frame in calling(beat):
                    yield frame
            case ToolCalls():
                made = True
            case Failed(message_text, _, spent):
                yield broke(spent, message_text)
                return
            case Finished(reason, usage):
                text = "".join(pieces)
                if checked is not None and not made:
                    try:
                        document(text, checked)
                    except (MalformedJSON, SchemaViolation) as violation:
                        # A stream cannot take back the text it already handed out, so the
                        # violation travels the way a generation that died travels.
                        yield broke(usage, failed(violation, checked.attempts)[0])
                        return
                for frame in message_done():
                    yield frame
                done = response(
                    request_id,
                    created,
                    request,
                    output,
                    "completed",
                    usage,
                    cut=reason == "length",
                )
                yield sse.sequenced("response.completed", next(sequence), {"response": done})
            case _:
                pass
