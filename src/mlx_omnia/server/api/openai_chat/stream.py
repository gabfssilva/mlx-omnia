"""Turning a read generation into this dialect's frames and bodies."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping

from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api import sse
from mlx_omnia.server.api.errors import openai_envelope
from mlx_omnia.server.api.openai_chat.codec import sequences
from mlx_omnia.server.api.openai_chat.models import ChatRequest
from mlx_omnia.server.api.responses import Checked, document, failed
from mlx_omnia.server.generation.collect import Completion
from mlx_omnia.server.generation.consume import Options, consume
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.runtime.events import (
    Failed,
    Finished,
    FinishReason,
    ReasoningDelta,
    Started,
    TextDelta,
    Timings,
    ToolCallDelta,
    ToolCalls,
    Usage,
)
from mlx_omnia.server.services import catalog


def reading_of(request: ChatRequest, job: Job, tools: bool) -> Options:
    """How this generation is read: the dialect's stop sequences, its budget, and the window
    the generation also ran under — a turn that cap cut is `length`, whatever was asked for."""
    return Options(
        tools=tools,
        stop=sequences(request.stop),
        max_tokens=request.max_tokens,
        context_limit=catalog.context_of(job.model_id),
    )


def finish_of(reason: FinishReason) -> str:
    """The dialect's three reasons. A call read whole outranks the budget: a client that gets
    one can execute it, and `length` beside an executable call sends it to render a truncation
    instead. A client's own sequence has no separate reason here — `stop` is what upstream
    reports, and the answer really did stop."""
    if reason == "tool_use":
        return "tool_calls"
    return "stop" if reason == "stop_sequence" else reason


def _chunk(
    request_id: str, created: int, model: str, delta: Mapping[str, object], finish: str | None
) -> str:
    return sse.data(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def usage_of(usage: Usage) -> dict[str, object]:
    """`prompt_tokens` is the whole prompt, reused or not: it is what the request sent. What
    the reuse changes rides under `prompt_tokens_details`, written even when it is zero so that
    a client can tell a miss from a server that does not carry the field."""
    return {
        "prompt_tokens": usage.prompt_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.reused_tokens},
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _timings(timings: Timings) -> dict[str, object]:
    """What the turn cost in seconds. The dialect has no field for any of it, so it rides under
    a prefixed key an SDK that does not know it drops — the same numbers `/admin/metrics`
    publishes, so a client reading the stream and a dashboard never disagree."""
    return {
        "load_seconds": timings.load_seconds,
        # Absent for a turn that did not speculate. Present, it is the one place the difference
        # shows: the rate alone never says a drafter was there.
        "speculation": (
            None
            if timings.speculation_rounds == 0
            else {
                "rounds": timings.speculation_rounds,
                "proposed": timings.speculation_proposed,
                "accepted": timings.speculation_accepted,
            }
        ),
        "ttft_seconds": timings.ttft_seconds,
        "prefill_tokens_per_second": timings.prefill_tokens_per_second,
        "tokens_per_second": timings.tokens_per_second,
        "bytes_per_token": timings.bytes_per_token,
        "ceiling_fraction": timings.ceiling_fraction,
    }


def _usage_chunk(request_id: str, created: int, model: str, usage: Usage, timings: Timings) -> str:
    """The extra frame `stream_options.include_usage` asks for: the whole request's usage and
    no choices, which is the shape the SDK folds into the completion it accumulates."""
    return sse.data(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage_of(usage),
            "x_mlx_omnia": _timings(timings),
        }
    )


def _call_frame(delta: ToolCallDelta) -> Mapping[str, object] | None:
    """The dialect's delta for what a reader just resolved.

    The first frame of a call carries `id`, `type` and the name with empty arguments; every
    frame after it carries only the fragment. That split is the SDK's own accumulator: it
    matches the entries of two frames by `index` — which is why every entry has one, and why it
    raises on an entry that has none — and it concatenates `function.arguments` across them, so
    a name repeated would be appended rather than replaced."""
    if delta.name is not None:
        return {
            "tool_calls": [
                {
                    "index": delta.index,
                    "id": f"call_{delta.id}",
                    "type": "function",
                    "function": {"name": delta.name, "arguments": delta.arguments},
                }
            ]
        }
    if not delta.arguments:
        # A delta that only says the call closed. The dialect has no frame for that —
        # `finish_reason` is what ends a call here — and an empty fragment would reach the
        # accumulator as a zero-length append.
        return None
    return {"tool_calls": [{"index": delta.index, "function": {"arguments": delta.arguments}}]}


def _error_frame(message: str, code: str) -> str:
    """A frame carrying `error` is what the SDK raises on, and the only way to fail a request
    that already answered 200. Without it a generation that died closes with
    `finish_reason: "stop"` and `[DONE]`, which is a completed answer."""
    return sse.data({"error": openai_envelope(message, code, "server_error")})


async def stream(
    job: Job,
    options: Options,
    request_id: str,
    created: int,
    model: str,
    usage: bool,
    checked: Checked | None,
) -> AsyncIterator[str]:
    sent: list[str] = []
    made = False
    yield _chunk(request_id, created, model, {"role": "assistant", "content": ""}, None)
    async for beat in consume(job, options):
        match beat:
            case None:
                yield sse.KEEP_ALIVE
            case Started():
                pass
            case ReasoningDelta(thought):
                # The field the dialect has for it, which is what DeepSeek, vLLM and Ollama
                # answer with over this same route. Without it the block goes out as the answer.
                yield _chunk(request_id, created, model, {"reasoning_content": thought}, None)
            case TextDelta(text):
                sent.append(text)
                yield _chunk(request_id, created, model, {"content": text}, None)
            case ToolCallDelta() as delta:
                if (frame := _call_frame(delta)) is not None:
                    yield _chunk(request_id, created, model, frame, None)
            case ToolCalls():
                made = True
            case Failed(message, code, _):
                yield _error_frame(message, code)
                return
            case Finished(reason, spent, _, timings):
                if checked is not None and not made:
                    try:
                        document("".join(sent), checked)
                    except (MalformedJSON, SchemaViolation) as violation:
                        # A stream cannot take back the text it already handed out, and a
                        # second attempt is refused for it: what is left is the door a
                        # generation that died goes through.
                        yield _error_frame(*failed(violation, 1))
                        return
                yield _chunk(request_id, created, model, {}, finish_of(reason))
                if usage:
                    # After the finish frame and before [DONE], where the dialect puts it.
                    yield _usage_chunk(request_id, created, model, spent, timings)
                yield sse.DONE


def message_of(completion: Completion) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": completion.text}
    if completion.reasoning:
        message["reasoning_content"] = completion.reasoning
    if completion.calls:
        # The dialect's null: a turn that only called something has no text, and `""` reads as
        # an assistant that answered with nothing.
        message["content"] = completion.text or None
        message["tool_calls"] = [
            {
                "index": call.index,
                "id": f"call_{call.id}",
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in completion.calls
        ]
    return message
