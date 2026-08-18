from mlx_omnia.server.api.responses.models import ResponsesRequest
from mlx_omnia.server.runtime.events import Usage


def part(text: str) -> dict[str, object]:
    return {"type": "output_text", "text": text, "annotations": []}


def message(message_id: str, text: str | None) -> dict[str, object]:
    """The text output item of a generation. `None` is the item the stream opens with —
    announced before any text exists, which is what gives the deltas an item to attach to."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "in_progress" if text is None else "completed",
        "content": [] if text is None else [part(text)],
    }


def call_item(item_id: str, call_id: str, name: str, arguments: str | None) -> dict[str, object]:
    """One call, as an output item of its own — which is this dialect's shape and not
    `chat/completions`'s list on the message. `None` is the item the stream opens with: the
    arguments arrive in the delta that follows."""
    return {
        "id": item_id,
        "type": "function_call",
        "status": "in_progress" if arguments is None else "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments or "",
    }


def _usage(usage: Usage) -> dict[str, object]:
    """`cache_write_tokens` stays at zero and is what it says: the trie is filled out of the
    forward this turn was going to run anyway, so there is nothing the client was charged for
    storing."""
    return {
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.reused_tokens,
            "cache_write_tokens": 0,
        },
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.total_tokens,
    }


def response(
    request_id: str,
    created: int,
    request: ResponsesRequest,
    output: list[dict[str, object]],
    status: str,
    usage: Usage | None,
    *,
    cut: bool = False,
    error: str | None = None,
) -> dict[str, object]:
    """The whole resource, which the stream carries twice: once empty and in progress at
    `response.created`, once complete at the end. The SDK builds its snapshot from the first
    and replaces it with the second, so both are the same shape.

    A generation the budget cut is `incomplete` and says why: `completed` with the text cut
    mid-sentence is what an agent loop reads as the final answer."""
    return {
        "id": request_id,
        "object": "response",
        "created_at": created,
        "status": "incomplete" if cut else status,
        "model": request.model,
        "output": output,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "parallel_tool_calls": False,
        "temperature": request.temperature,
        "tool_choice": request.tool_choice,
        "tools": [tool.model_dump(exclude_none=True) for tool in request.tools or ()],
        "top_p": request.top_p,
        "metadata": {},
        "incomplete_details": {"reason": "max_output_tokens"} if cut else None,
        "usage": None if usage is None else _usage(usage),
        "error": None if error is None else {"code": "server_error", "message": error},
    }
