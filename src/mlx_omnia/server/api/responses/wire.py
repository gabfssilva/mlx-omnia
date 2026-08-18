import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlx_omnia import Chat, ImagePart, TextPart, ToolCallRequest
from mlx_omnia import ChatMessage as Turn
from mlx_omnia.engine.schema import (
    MalformedJSON,
    SchemaViolation,
    extract_json,
    json_instruction,
    validate,
)


class UnreadableArguments(ValueError):
    """A call the client replayed whose arguments are not a JSON object. The message is the
    client's — each dialect puts it in its own envelope."""


def called(id: str, name: str, arguments: str) -> ToolCallRequest:
    """One replayed call, from the JSON **text** two dialects put it in to the value the
    templates read.

    Of the templates in circulation that declare tools, eight read only a mapping
    (`arguments|items`) and none reads only the text — so the text is a fact of two wire
    formats and not of a conversation, and it stops being one here.
    """
    try:
        payload = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as error:
        raise UnreadableArguments(
            f"the arguments of the call to {name!r} are not JSON ({error.msg})"
        ) from error
    if not isinstance(payload, dict):
        raise UnreadableArguments(f"the arguments of the call to {name!r} are not a JSON object")
    named: dict[str, object] = {str(key): value for key, value in payload.items()}
    return {"id": id, "type": "function", "function": {"name": name, "arguments": named}}


def content_of(
    parts: Sequence[TextPart | ImagePart],
) -> str | tuple[TextPart | ImagePart, ...]:
    """A turn's content as the template will read it: the characters when there is no image in
    it, the parts themselves when there is.

    Not a list either way, because a template concatenates `content` — `{{ message['content'] }}`
    over a one-element list writes the list's repr into the prompt — and only the vision branch
    of the templates in circulation iterates it.
    """
    texts = [part["text"] for part in parts if part["type"] == "text"]
    return "".join(texts) if len(texts) == len(parts) else tuple(parts)


def unsupported_reason(model: str, conversation: Chat) -> str:
    """Why a model took no conversation, in words a client can act on. The image is named
    first: for a checkpoint with neither template nor vision tower both lines are true, and the
    one about what the client just attached is the useful one."""
    for message in conversation.messages:
        content = message["content"]
        if not isinstance(content, str) and any(part["type"] == "image" for part in content):
            return f"model {model!r} does not accept an image: the checkpoint has no vision tower"
    return f"model {model!r} does not accept a conversation: the checkpoint ships no chat template"


@dataclass(frozen=True)
class Checked:
    """What one request asks to be checked: the schema the answer is measured against — `None`
    is `json_object`, which asks only that the answer be JSON — and how many whole generations
    the client agreed to pay for."""

    schema: Mapping[str, object] | None
    attempts: int


def instruction(schema: Mapping[str, object] | None) -> Turn:
    """The schema as the model reads it: a turn like any other, and the last one. The text is
    the engine's `json_instruction`, so what is asked for and what `validate` enforces cannot
    drift into two readings."""
    return {"role": "system", "content": json_instruction(schema)}


def document(content: str, checked: Checked) -> object:
    """The JSON value the answer carries, against the schema when there is one. Raises
    `MalformedJSON` or `SchemaViolation`."""
    value = extract_json(content)
    if checked.schema is not None:
        validate(value, checked.schema)
    return value


def failed(failure: MalformedJSON | SchemaViolation, attempts: int) -> tuple[str, str]:
    """What the client is told about an answer that did not validate, and under which code.
    The generations spent are in the message: what this level costs is interactions."""
    spent = f"after {attempts} generation{'' if attempts == 1 else 's'}"
    if isinstance(failure, SchemaViolation):
        where = f"{failure.path} {failure.reason}"
        message = f"the answer does not validate against the schema {spent}: {where}"
        return message, "schema_violation"
    return f"the answer carries no JSON value {spent}", "malformed_json"


def declared(
    name: str, description: str | None, parameters: Mapping[str, object] | None
) -> Mapping[str, object]:
    """One function offered to the model, in the nested envelope every template reads —
    `tool.function.name` — whatever shape the dialect spelled it in.

    The keys go in in this order because a template that renders the entry with `tojson` writes
    them in it: the same function declared through two dialects has to reach the model as the
    same characters, or the two prompts are two prompts.
    """
    function: dict[str, object] = {"name": name}
    if description is not None:
        function["description"] = description
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}
