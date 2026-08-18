from collections.abc import Mapping

from mlx_omnia import ChatMessage as Turn
from mlx_omnia import ImagePart, TextPart
from mlx_omnia.server.api.responses.models import (
    ContentPart,
    FunctionCallItem,
    FunctionOutputItem,
    InputImage,
    InputItem,
    JsonObjectOutput,
    MessageItem,
    ResponsesRequest,
    SchemaOutput,
    TextOutput,
)
from mlx_omnia.server.api.responses.png import inline_image
from mlx_omnia.server.api.responses.wire import Checked, called, content_of, declared


def _given_part(part: ContentPart | InputImage) -> TextPart | ImagePart:
    if isinstance(part, ContentPart):
        return {"type": "text", "text": part.text}
    return inline_image(part.image_url)


def _turn(item: MessageItem) -> Turn:
    """`developer` is the dialect's newer name for a system turn, and the template knows only
    the older one. The text parts of a content list concatenate: what separates them is a
    boundary in the client's own structure, not text the model should read."""
    content = (
        item.content
        if isinstance(item.content, str)
        else content_of([_given_part(part) for part in item.content])
    )
    return {
        "role": "system" if item.role == "developer" else item.role,
        "content": content,
    }


def given(input: str | list[InputItem]) -> tuple[Turn, ...]:
    """What the client sent, as turns: a bare string is the user message it stands for.

    A call is an item of its own here and a key of the assistant's turn in every template, so
    consecutive calls fold into the one turn that made them — two turns would tell the model it
    answered twice.
    """
    if isinstance(input, str):
        return ({"role": "user", "content": input},)
    turns: list[Turn] = []
    for item in input:
        match item:
            case MessageItem():
                turns.append(_turn(item))
            case FunctionCallItem():
                call = called(item.call_id, item.name, item.arguments)
                previous = turns[-1] if turns else None
                if previous is not None and previous["role"] == "assistant":
                    # The canonical replay is `input + response.output`, and a generation that
                    # wrote text *and* called something is `[message, function_call]` — one
                    # turn of the model, which two assistant turns would tell it was two.
                    previous["tool_calls"] = [*previous.get("tool_calls", []), call]
                else:
                    turns.append({"role": "assistant", "content": "", "tool_calls": [call]})
            case FunctionOutputItem():
                turns.append(
                    {
                        "role": "tool",
                        "content": item.output,
                        "tool_call_id": item.call_id,
                    }
                )
    return tuple(turns)


def tools(request: ResponsesRequest) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: "none"` is honoured where it can be honoured: the tools never enter the
    prompt, so the model has nothing to call rather than an instruction not to."""
    if request.tools is None or request.tool_choice == "none":
        return ()
    return tuple(declared(tool.name, tool.description, tool.parameters) for tool in request.tools)


def _wanted(request: ResponsesRequest) -> JsonObjectOutput | SchemaOutput | None:
    """What this request asks of the answer, or `None` when it asks for nothing."""
    text = request.text
    asked = None if text is None else text.format
    return None if isinstance(asked, TextOutput) else asked


def guaranteed(request: ResponsesRequest) -> Mapping[str, object] | None:
    """The schema this request asks to be *guaranteed* rather than checked, or `None` for every
    other shape."""
    wanted = _wanted(request)
    return wanted.definition if isinstance(wanted, SchemaOutput) and wanted.strict else None


def checked_of(request: ResponsesRequest) -> Checked | None:
    """At most one of the two levels: with `strict` the grammar makes a violation unreachable
    and there is nothing left to check afterwards."""
    wanted = _wanted(request)
    if guaranteed(request) is not None or wanted is None:
        return None
    return Checked(wanted.definition if isinstance(wanted, SchemaOutput) else None, 1)


def prefixed(
    given: tuple[Turn, ...], instructions: str | None, system_prompt: str | None
) -> tuple[Turn, ...]:
    """The profile's system prompt goes in only when nothing else claimed the place — what
    keeps the template from rendering two system turns for the model to pick between."""
    if instructions is not None:
        given = ({"role": "system", "content": instructions}, *given)
    if system_prompt is None or any(turn["role"] == "system" for turn in given):
        return given
    return ({"role": "system", "content": system_prompt}, *given)
