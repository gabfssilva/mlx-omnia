"""The translation between this dialect's wire and the engine's conversation, both ways."""

from collections.abc import Mapping

from fastapi.responses import JSONResponse

from mlx_omnia import (
    Chat,
    GenerationOptions,
    ImagePart,
    LogitFilter,
    TextPart,
    ToolCallRequest,
    greedy,
    min_p,
    repetition_penalty,
    sampler,
    temperature,
    top_k,
    top_p,
)
from mlx_omnia import ChatMessage as Turn
from mlx_omnia.engine.chat import Effort
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.server.api.anthropic.models import (
    Conversation,
    ImageBlock,
    Message,
    MessagesRequest,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mlx_omnia.server.api.errors import anthropic_error
from mlx_omnia.server.api.responses import content_of, declared, image_part
from mlx_omnia.server.services.profiles import Sampling


def encode_error(status: int, kind: str, message: str) -> JSONResponse:
    """The dialect's envelope, which the app's validation handler also writes for a refused
    body under this prefix."""
    return anthropic_error(status, kind, message)


def _text(content: str | list[TextBlock]) -> str:
    return content if isinstance(content, str) else "".join(block.text for block in content)


_BILLING = "x-anthropic-billing-header:"


def _system(content: str | list[TextBlock]) -> str:
    """The system prompt, without the block that is not one.

    Claude Code opens `system` with `x-anthropic-billing-header: …` on every request. It is
    metadata for a server this is not, and it sits at the front of the prefix every request
    reuses.
    """
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if not block.text.startswith(_BILLING))


def _called(block: ToolUseBlock) -> ToolCallRequest:
    """The call in the shape the templates read: nested under `function`, arguments as the
    mapping they already are on this wire."""
    return {
        "id": block.id,
        "type": "function",
        "function": {"name": block.name, "arguments": block.input},
    }


def _part(block: TextBlock | ImageBlock) -> TextPart | ImagePart:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return image_part(block.source.data, block.source.media_type)


def _turns(message: Message) -> list[Turn]:
    """One message, and the turns it spells. Its results come first because they are the round
    the message answers; the text and image blocks keep the order they arrived in, because
    where an image sits among the words is what the template writes a marker for."""
    content = message.content
    if isinstance(content, str):
        return [{"role": message.role, "content": content}]
    said = content_of(
        [_part(block) for block in content if isinstance(block, TextBlock | ImageBlock)]
    )
    made = [_called(block) for block in content if isinstance(block, ToolUseBlock)]
    results = [block for block in content if isinstance(block, ToolResultBlock)]
    turns: list[Turn] = [
        {"role": "tool", "content": _text(block.content), "tool_call_id": block.tool_use_id}
        for block in results
    ]
    if said or made or not results:
        turn: Turn = {"role": message.role, "content": said}
        if made:
            turn["tool_calls"] = made
        turns.append(turn)
    return turns


def declared_tools(request: Conversation) -> tuple[Mapping[str, object], ...]:
    """`tool_choice: {"type": "none"}` is honoured where it can be: the tools never enter the
    prompt, so the model has nothing to call rather than an instruction not to."""
    choice = request.tool_choice
    if request.tools is None or (choice is not None and choice.type == "none"):
        return ()
    return tuple(declared(tool.name, tool.description, tool.input_schema) for tool in request.tools)


def _thinks(request: Conversation, preset: Effort | None) -> Effort:
    """A request that names `thinking` throws the switch and nothing more. One that names none
    falls to the profile, and then to `auto` — the template's own default and not `off`."""
    thinking = request.thinking
    if thinking is None:
        return "auto" if preset is None else preset
    return "off" if thinking.type == "disabled" else "on"


def reasoning_shown(request: Conversation) -> bool:
    """Whether the reasoning reaches the client as text. A request that named no `thinking`
    gets it; one that named `omitted` chose upstream's default."""
    thinking = request.thinking
    return True if thinking is None else thinking.display == "summarized"


def to_conversation(
    request: Conversation, preset: str | None, effort: Effort | None = None
) -> Chat:
    """The translation this dialect exists for: `system` is a field on the way in and the first
    turn on the way out. The profile's prompt fills it only when the request left it out, and a
    system that comes out empty opens no turn."""
    system = _system(request.system) if request.system is not None else preset
    turns: list[Turn] = [] if not system else [{"role": "system", "content": system}]
    for message in request.messages:
        turns += _turns(message)
    return Chat(
        tuple(turns), tools=declared_tools(request), reasoning_effort=_thinks(request, effort)
    )


def generation_options(
    request: MessagesRequest, sampling: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """The preset fills the knobs the client left out, and only those. Which ones it left out
    is `model_fields_set`: the dialect's defaults are values like any other, so an unset field
    cannot be told from an explicit one by its value. `min_p`, `repetition_penalty` and `seed`
    have no field here at all, so a profile is the only thing that can set them."""
    asked = request.model_fields_set
    heat = (
        request.temperature
        if "temperature" in asked or sampling.temperature is None
        else sampling.temperature
    )
    nucleus = request.top_p if "top_p" in asked or sampling.top_p is None else sampling.top_p
    kept = request.top_k if "top_k" in asked or sampling.top_k is None else sampling.top_k
    repeats = sampling.repetition_penalty
    penalty = None if repeats is None else repetition_penalty(repeats)
    thinking = request.thinking
    budget = None if thinking is None else thinking.budget_tokens
    if budget is None:
        budget = sampling.reasoning_budget
    if heat == 0.0:
        # The deterministic end of the dial: no distribution is left to draw from, and dividing
        # by it would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=request.max_tokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=budget,
        )

    filters: list[LogitFilter] = [temperature(heat)]
    if kept is not None:
        filters.append(top_k(kept))
    if nucleus is not None:
        filters.append(top_p(nucleus))
    if sampling.min_p is not None:
        filters.append(min_p(sampling.min_p))
    return GenerationOptions(
        max_tokens=request.max_tokens,
        sampler=sampler(*filters, seed=sampling.seed),
        penalty=penalty,
        constraint=constraint,
        reasoning_budget=budget,
    )
