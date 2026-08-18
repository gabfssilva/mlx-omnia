import json
import time
import uuid
from dataclasses import replace

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omnia import Chat, UnsupportedInput
from mlx_omnia import ChatMessage as Turn
from mlx_omnia.engine.grammar import GrammarRefused
from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api import sse
from mlx_omnia.server.api.errors import openai_error
from mlx_omnia.server.api.responses.frames import call_item, message, response
from mlx_omnia.server.api.responses.inputs import (
    checked_of,
    guaranteed,
    prefixed,
)
from mlx_omnia.server.api.responses.inputs import (
    given as given_of,
)
from mlx_omnia.server.api.responses.inputs import (
    tools as tools_of,
)
from mlx_omnia.server.api.responses.models import ResponsesRequest
from mlx_omnia.server.api.responses.png import UnreadableImage
from mlx_omnia.server.api.responses.sampling import effort_of, options, preset_of
from mlx_omnia.server.api.responses.stream import dialect, frames
from mlx_omnia.server.api.responses.wire import (
    Checked,
    UnreadableArguments,
    document,
    failed,
    instruction,
    unsupported_reason,
)
from mlx_omnia.server.deps import EngineDep
from mlx_omnia.server.generation.collect import collect
from mlx_omnia.server.runtime.engine import NotConstrainable, NotQuantizable
from mlx_omnia.server.services import profiles
from mlx_omnia.server.services.profiles import ProfileView, Sampling

router = APIRouter()


def _refusal(request: ResponsesRequest) -> JSONResponse | None:
    """The three bodies this route cannot honour, in the order they are read."""
    if request.store:
        return openai_error(
            400,
            "store is not supported: this server keeps no responses, so an id it handed "
            "back would name nothing",
            "store_unsupported",
        )
    if any(tool.strict for tool in request.tools or ()):
        return openai_error(
            400,
            "strict on a tool is not supported: what a grammar constrains here is the whole "
            "answer (text.format), not the arguments of a call, so an argument that violates "
            "the schema would come back as one that was checked against it",
            "strict_unsupported",
        )
    if guaranteed(request) is not None and tools_of(request):
        return openai_error(
            400,
            "a strict text.format and tools cannot both be honoured: the grammar constrains "
            "decoding to the schema from the first token, so the model cannot write a call "
            "however it is offered. Send the same format without strict, or offer no tools.",
            "strict_with_tools",
        )
    return None


def _conversation(
    request: ResponsesRequest,
    given: tuple[Turn, ...],
    profile: ProfileView | None,
    preset: Sampling,
) -> tuple[Chat, Checked | None]:
    turns = prefixed(
        given,
        request.instructions,
        None if profile is None else profile.system_prompt,
    )
    checked = checked_of(request)
    if checked is not None:
        turns = (*turns, instruction(checked.schema))
    reasoning = request.reasoning
    effort = effort_of(None if reasoning is None else reasoning.effort, preset.reasoning_effort)
    return Chat(turns, tools=tools_of(request), reasoning_effort=effort), checked


@router.post("/api/openai/v1/responses", response_model=None)
async def respond(request: ResponsesRequest, engine: EngineDep) -> JSONResponse | StreamingResponse:
    if (refused := _refusal(request)) is not None:
        return refused
    try:
        given = given_of(request.input)
    except UnreadableImage as unreadable:
        return openai_error(400, str(unreadable), "invalid_image")
    except UnreadableArguments as unreadable:
        return openai_error(400, str(unreadable), "invalid_tool_arguments")
    if not any(turn["content"] or turn.get("tool_calls") for turn in given):
        return openai_error(400, "input must contain non-empty text", "empty_input")

    model_id, profile = await profiles.resolve(request.model)
    preset = await profiles.preset(model_id, profile)
    # `asked` and not `request` is what the answer echoes: the knobs a profile filled are the
    # ones that generated the text, and a client reading its own back learns nothing.
    asked = preset_of(request, preset)
    conversation, checked = _conversation(asked, given, profile, preset)
    strict = guaranteed(asked)
    request_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        constrained = None if strict is None else await engine.constrain(model_id, strict)
        job = await engine.submit(
            model_id,
            conversation,
            replace(
                options(
                    asked,
                    preset,
                    constrained,
                    max_tokens=asked.max_output_tokens,
                    context_limit=None,
                ),
                speculate=await profiles.speculating(model_id, profile),
            ),
        )
    except GrammarRefused as refusal:
        return openai_error(400, str(refusal), "grammar_refused")
    except NotConstrainable as refusal:
        return openai_error(400, str(refusal), "not_constrainable")
    except NotQuantizable as refusal:
        return openai_error(400, str(refusal), "not_quantizable")
    except UnsupportedInput:
        return openai_error(
            400, unsupported_reason(request.model, conversation), "unsupported_input"
        )
    except Exception as error:
        return openai_error(
            404, f"model {request.model!r} is not available: {error}", "model_not_found"
        )

    reading = dialect(asked, bool(tools_of(asked)))
    if request.stream:
        return StreamingResponse(
            frames(
                job,
                asked,
                request_id=request_id,
                message_id=message_id,
                created=created,
                checked=checked,
                options=reading,
            ),
            media_type=sse.MEDIA_TYPE,
        )

    answer = await collect(job, reading)
    if answer.failure is not None:
        return openai_error(500, answer.failure.message, answer.failure.code, kind="server_error")
    content = answer.text
    # A turn that called something answered with a call and not with a document: the format is
    # about the answer, and this turn has none to check.
    if checked is not None and not answer.calls:
        try:
            value = document(content, checked)
        except (MalformedJSON, SchemaViolation) as violation:
            reason, code = failed(violation, checked.attempts)
            # 422 and not a 5xx: the SDKs retry a 5xx on their own, and a whole generation
            # bought behind the client's back is the cost this level exists to make visible.
            return openai_error(422, reason, code, kind="server_error")
        # The client asked for a JSON value, and prose or a fence with one inside it is an
        # answer `json.loads` refuses. What goes back is what was validated.
        content = json.dumps(value, ensure_ascii=False)
    # A turn that only called something has no message item: an empty one is an assistant that
    # answered with nothing, which is not what happened.
    written = not answer.calls or bool(content)
    output: list[dict[str, object]] = [message(message_id, content)] if written else []
    output += [
        call_item(
            f"fc_{call.id}",
            f"call_{call.id}",
            call.name,
            json.dumps(call.arguments),
        )
        for call in answer.calls
    ]
    return JSONResponse(
        content=response(
            request_id,
            created,
            asked,
            output,
            "completed",
            answer.usage,
            cut=answer.reason == "length",
        )
    )
