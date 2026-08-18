"""`POST /api/openai/v1/chat/completions` and `GET /api/openai/v1/models`."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omnia import Chat, UnsupportedInput
from mlx_omnia.engine.chat import Effort
from mlx_omnia.engine.grammar import GrammarRefused
from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api import sse
from mlx_omnia.server.api.errors import openai_error
from mlx_omnia.server.api.openai_chat.codec import (
    checked_of,
    correction,
    envelope_of,
    forced,
    messages_of,
    refusal_for,
    strict_of,
    tools_of,
    turn_of,
)
from mlx_omnia.server.api.openai_chat.models import ChatRequest
from mlx_omnia.server.api.openai_chat.stream import (
    finish_of,
    message_of,
    reading_of,
    stream,
    usage_of,
)
from mlx_omnia.server.api.responses import (
    UnreadableArguments,
    UnreadableImage,
    document,
    effort_of,
    failed,
    instruction,
    options,
    preset_of,
    unsupported_reason,
)
from mlx_omnia.server.generation.collect import collect
from mlx_omnia.server.runtime.engine import Engine, Job, NotConstrainable, NotQuantizable
from mlx_omnia.server.runtime.events import Usage
from mlx_omnia.server.services import catalog, profiles
from mlx_omnia.server.services.profiles import ProfileView, Sampling


async def _engine_of(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine_of)]

router = APIRouter()


@router.get("/api/openai/v1/models")
async def models() -> dict[str, object]:
    """The catalog, not the residents. No dialect's schema has a notion of loaded, and a client
    cannot ask for a model it cannot see — a list of residents is empty exactly at boot, which
    is when it is read. Which of them are in memory is `/admin`'s question.

    A model with profiles answers to more than one name, and every one of them is listed: the
    dialect has no field for a profile, so an id a client cannot see is a profile it cannot
    select."""
    return {
        "object": "list",
        "data": [
            {"id": served, "object": "model", "created": 0, "owned_by": "mlx_omnia"}
            for served in await profiles.served_ids()
        ],
    }


async def _submitted(
    engine: Engine,
    model_id: str,
    profile: ProfileView | None,
    request: ChatRequest,
    asked: ChatRequest,
    conversation: Chat,
    tools: tuple[Mapping[str, object], ...],
    strict: Mapping[str, object] | None,
    sampling: Sampling,
) -> Job | JSONResponse:
    """This attempt's generation, or the refusal it earned. A name that does not resolve to a
    checkpoint and one whose load fails are the same answer to the client: this model is not
    available here."""
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this attempt's.
        constrained = None if strict is None else await engine.constrain(model_id, strict)
        if forced(asked):
            # The family of this checkpoint's own template, read off the loaded model rather
            # than guessed.
            envelope = envelope_of(await engine.reachable(model_id))
            if envelope is None:
                return openai_error(
                    400,
                    'tool_choice: "required" cannot be honoured for this checkpoint: forcing a '
                    "call constrains decoding to its call envelope, and this model's family "
                    'does not express one exactly. Send tool_choice: "auto", or use a '
                    "checkpoint whose calls are JSON.",
                    "tool_choice_unsupported",
                )
            constrained = await engine.constrain_envelope(model_id, envelope, tools)
        return await engine.submit(
            model_id,
            conversation,
            replace(
                options(
                    asked,
                    sampling,
                    constrained,
                    max_tokens=asked.max_tokens,
                    context_limit=catalog.context_of(request.model),
                ),
                speculate=await profiles.speculating(model_id, profile),
            ),
        )
    except GrammarRefused as refusal:
        # The compiler's own words: what the client does with them is send the same schema
        # without `strict` and have the answer checked instead of guaranteed.
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


@router.post("/api/openai/v1/chat/completions", response_model=None)
async def chat(request: ChatRequest, engine: EngineDep) -> JSONResponse | StreamingResponse:
    if not any(message.content for message in request.messages):
        return openai_error(400, "messages must contain non-empty content", "empty_prompt")
    if (refused := refusal_for(request)) is not None:
        return refused

    model_id, profile = await profiles.resolve(request.model)
    preset = await profiles.preset(model_id, profile)
    asked = preset_of(request, preset)
    messages = request.messages if profile is None else messages_of(request, profile.system_prompt)
    tools = tools_of(request)
    checked = checked_of(request)
    strict = strict_of(request)
    # The conversation goes to the model as a conversation: what turns it into a prompt is the
    # checkpoint's own chat template, and a model that ships none says so below.
    try:
        turns = tuple(turn_of(message) for message in messages)
    except UnreadableImage as unreadable:
        return openai_error(400, str(unreadable), "invalid_image")
    except UnreadableArguments as unreadable:
        # 400 and not a 500: the client replayed a call whose arguments are not a JSON object,
        # and the templates read a mapping.
        return openai_error(400, str(unreadable), "invalid_tool_arguments")
    if checked is not None:
        turns = (*turns, instruction(checked.schema))
    effort: Effort = effort_of(request.reasoning_effort, preset.reasoning_effort)
    conversation = Chat(turns, tools=tools, reasoning_effort=effort)
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    spent = Usage()
    attempt = 0
    while True:
        attempt += 1
        submitted = await _submitted(
            engine,
            model_id,
            profile,
            request,
            asked,
            conversation,
            tools,
            strict,
            preset,
        )
        if isinstance(submitted, JSONResponse):
            return submitted
        job = submitted
        reading = reading_of(asked, job, bool(tools))

        if request.stream:
            # The one pass a stream gets: whatever is checked is checked at the end of it, and
            # there is no second attempt to come back here for.
            streaming = request.stream_options
            return StreamingResponse(
                stream(
                    job,
                    reading,
                    request_id,
                    created,
                    request.model,
                    streaming is not None and streaming.include_usage,
                    checked,
                ),
                media_type=sse.MEDIA_TYPE,
            )

        completion = await collect(job, reading)
        if completion.failure is not None:
            return openai_error(
                500, completion.failure.message, "generation_failed", kind="server_error"
            )
        # Every attempt is a whole generation and the client paid for all of them: usage that
        # counted only the one that validated would disagree with `schema_attempts`.
        spent = spent + completion.usage
        if checked is None or completion.calls:
            # A turn that called something answered with a call and not with a document.
            break
        try:
            value = document(completion.text, checked)
        except (MalformedJSON, SchemaViolation) as violation:
            if attempt >= checked.attempts:
                reason, code = failed(violation, attempt)
                # 422 and not a 5xx: the SDKs retry a 5xx on their own, and a whole generation
                # bought behind the client's back is what this level makes visible.
                return openai_error(422, reason, code, kind="server_error")
            conversation = Chat((*turns, *correction(completion.text, violation)), tools=tools)
            continue
        # The document and not the text around it: the client asked for a JSON value, and prose
        # or a fence with one inside it is an answer `json.loads` refuses.
        completion = replace(completion, text=json.dumps(value, ensure_ascii=False))
        break

    body: dict[str, object] = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message_of(completion),
                "finish_reason": finish_of(completion.reason),
            }
        ],
        "usage": usage_of(spent),
    }
    if checked is not None:
        # What the answer cost in generations, always — the default included.
        body["schema_attempts"] = attempt
    return JSONResponse(content=body)
