"""The codec between the dialect's wire models and the engine's `Chat`."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi.responses import JSONResponse

from mlx_omnia import ChatMessage as Turn
from mlx_omnia import ImagePart, TextPart
from mlx_omnia.engine.chat import parser_of
from mlx_omnia.engine.parsers import ToolFamily
from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api.errors import openai_error
from mlx_omnia.server.api.openai_chat.models import (
    ChatMessage,
    ChatRequest,
    JsonSchemaFormat,
    Part,
    TextContent,
    TextFormat,
)
from mlx_omnia.server.api.responses import (
    Checked,
    called,
    content_of,
    declared,
    inline_image,
)


def messages_of(request: ChatRequest, system_prompt: str | None) -> list[ChatMessage]:
    """The profile's system prompt is the conversation's first turn, unless the client sent one
    of its own — which keeps the template from rendering two system turns to pick between."""
    if system_prompt is None or any(message.role == "system" for message in request.messages):
        return request.messages
    return [ChatMessage(role="system", content=system_prompt), *request.messages]


def _part(part: Part) -> TextPart | ImagePart:
    return (
        {"type": "text", "text": part.text}
        if isinstance(part, TextContent)
        else inline_image(part.image_url.url)
    )


def _content(content: str | list[Part] | None) -> str | tuple[TextPart | ImagePart, ...]:
    """What the turn carries, in the shape the template reads. `None` is `''` — the assistant
    turn that only called something, whose content the template concatenates."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return content_of([_part(part) for part in content])


def turn_of(message: ChatMessage) -> Turn:
    """A call is decoded on the way in, not handed through: this dialect spells `arguments` as
    JSON text and the templates read a mapping."""
    turn: Turn = {"role": message.role, "content": _content(message.content)}
    if message.tool_calls is not None:
        turn["tool_calls"] = [
            called(call.id, call.function.name, call.function.arguments)
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        turn["tool_call_id"] = message.tool_call_id
    return turn


def tools_of(request: ChatRequest) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: "none"` is honoured where it can be honoured: the tools never enter the
    prompt, so the model has nothing to call rather than an instruction not to.

    `declared` and not this dialect's own `model_dump`: a template that renders the entry with
    `tojson` writes the keys in the order they are in, so the same function declared through
    two dialects has to be built in one place or the two prompts differ."""
    if request.tools is None or request.tool_choice == "none":
        return ()
    return tuple(
        declared(tool.function.name, tool.function.description, tool.function.parameters)
        for tool in request.tools
    )


def checked_of(request: ChatRequest) -> Checked | None:
    """What this request asks to be checked, or `None` when it asks for nothing."""
    wanted = request.response_format
    if wanted is None or isinstance(wanted, TextFormat):
        return None
    schema = wanted.json_schema.definition if isinstance(wanted, JsonSchemaFormat) else None
    return Checked(schema, request.max_schema_attempts)


def envelope_of(model: object) -> ToolFamily | None:
    """This checkpoint's tool family when it can express its envelope as a grammar, and `None`
    otherwise — which is what a forced `tool_choice` is refused on."""
    parser = parser_of(model)
    family = None if parser is None else parser.tools
    return family if family is not None and family.grammar is not None else None


def forced(request: ChatRequest) -> bool:
    return request.tool_choice == "required" and bool(request.tools)


def strict_of(request: ChatRequest) -> Mapping[str, object] | None:
    """The schema this request asks to be *guaranteed* rather than checked. Under it decoding
    is constrained, so there is no answer that violates the schema and no attempt to buy."""
    wanted = request.response_format
    if not isinstance(wanted, JsonSchemaFormat) or not wanted.json_schema.strict:
        return None
    return wanted.json_schema.definition


def refusal_for(request: ChatRequest) -> JSONResponse | None:
    """The shapes this route cannot honour, each named."""
    if request.tool_choice == "required" and not request.tools:
        return openai_error(
            400,
            'tool_choice: "required" asks the model to call something and this request offers '
            "no tools: the constraint has no set of functions to pin the call to.",
            "tool_choice",
        )
    wanted = request.response_format
    if (
        isinstance(wanted, JsonSchemaFormat)
        and wanted.json_schema.strict
        and wanted.json_schema.definition is None
    ):
        return openai_error(
            400,
            "strict promises a decode that cannot violate the schema, and this request carries "
            "no schema: json_schema holds it under `schema`, and it is what decoding is "
            "constrained by.",
            "strict_without_schema",
        )
    if strict_of(request) is not None and tools_of(request):
        return openai_error(
            400,
            "strict and tools cannot both be honoured: the grammar constrains decoding to the "
            "schema from the first token, so the model cannot write a call however it is "
            "offered. Send the same json_schema without strict, or offer no tools.",
            "strict_with_tools",
        )
    if checked_of(request) is None:
        if "max_schema_attempts" in request.model_fields_set:
            return openai_error(
                400,
                "max_schema_attempts counts the generations spent making an answer validate, "
                "and this request asks for nothing to be validated: it needs a response_format.",
                "max_schema_attempts",
            )
        return None
    if request.stream and request.max_schema_attempts > 1:
        return openai_error(
            400,
            "max_schema_attempts above 1 cannot be honoured with stream: a second attempt is a "
            "second generation, and the frames of the first one have already left.",
            "stream_attempts",
        )
    return None


def correction(output: str, failure: MalformedJSON | SchemaViolation) -> tuple[Turn, Turn]:
    """What the next attempt reads: what the model wrote, and what was wrong with it.

    A retry over the same turns is the same generation — greedy decoding lands on the same
    tokens — so what makes the second attempt a second attempt is these two turns."""
    reason = (
        f"{failure.path} {failure.reason}"
        if isinstance(failure, SchemaViolation)
        else "it carried no JSON value"
    )
    return (
        {"role": "assistant", "content": output},
        {
            "role": "user",
            "content": f"That answer was not accepted: {reason}. Answer again, corrected, "
            "with the JSON value and nothing else.",
        },
    )


def sequences(stop: str | list[str] | None) -> list[str]:
    """The dialect spells one sequence as a bare string and several as a list."""
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else stop
