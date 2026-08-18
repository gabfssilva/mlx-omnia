"""The routes: `/api/gemini/v1beta/models/{model}:{method}`.

The method rides in the path, glued to the model name by a colon, so the whole tail is the
route and the split happens here: an id of this house carries a `/` and may carry a `:` (the
profile suffix), and only the *last* colon is the method's."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omnia import Chat, UnsupportedInput
from mlx_omnia.engine.grammar import GrammarRefused
from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api.errors import gemini_envelope, gemini_error
from mlx_omnia.server.api.gemini.codec import (
    call_part,
    dialect_options,
    effort_of,
    generation_options,
    refusal_of,
    reply,
    tools_of,
    turns_of,
    violation_reason,
)
from mlx_omnia.server.api.gemini.models import GenerateRequest, GenerationConfig
from mlx_omnia.server.api.responses import (
    Checked,
    UnreadableImage,
    document,
    instruction,
    unsupported_reason,
)
from mlx_omnia.server.api.sse import MEDIA_TYPE, data
from mlx_omnia.server.generation.collect import collect
from mlx_omnia.server.generation.consume import Options, consume
from mlx_omnia.server.runtime.engine import Engine, Job, NotConstrainable, NotQuantizable
from mlx_omnia.server.runtime.events import Failed, Finished, TextDelta, ToolCalls
from mlx_omnia.server.runtime.events import ToolCall as Called
from mlx_omnia.server.services import profiles

_METHODS = ("generateContent", "streamGenerateContent")


async def _engine_of(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine_of)]


async def _frames(
    job: Job, model: str, options: Options, checked: Checked | None
) -> AsyncIterator[str]:
    """The generation as this dialect's frames. The calls ride the closing frame, whole and all
    of them: `functionCall` has no partial form in this API, and the frame that already carries
    the finish reason is the one a client reads a completed turn off.

    A call read whole comes with a text tail the reader was holding, so with functions offered
    the last text seen waits for the frame after it rather than going out on its own."""
    sent: list[str] = []
    pending = ""
    made: tuple[Called, ...] = ()
    async for beat in consume(job, options):
        match beat:
            case TextDelta(text):
                sent.append(text)
                if not options.tools:
                    yield data(reply(model, [{"text": text}], None))
                    continue
                if pending:
                    yield data(reply(model, [{"text": pending}], None))
                pending = text
            case ToolCalls(calls):
                made = calls
            case Failed(message):
                # The SDK reads a chunk that opens with `error` and raises on it, which is the
                # only way to fail a generation that already answered 200. Without this the
                # closing frame would carry `STOP` and a `usageMetadata` counted off a
                # generation that died.
                yield data(gemini_envelope("INTERNAL", message))
                return
            case Finished(reason, usage):
                if checked is not None and not made:
                    try:
                        document("".join(sent), checked)
                    except (MalformedJSON, SchemaViolation) as violation:
                        reason_text = violation_reason(violation, checked)
                        yield data(gemini_envelope("INTERNAL", reason_text))
                        return
                parts: list[dict[str, object]] = [{"text": pending}] if pending else []
                parts += [call_part(call) for call in made]
                yield data(reply(model, parts or [{"text": ""}], (reason, usage)))
            case _:
                pass


router = APIRouter()


@router.get("/api/gemini/v1beta/models")
async def models() -> dict[str, object]:
    """The catalog under this dialect's name for a model — `models/{id}`, which is what the SDK
    puts back into the path. Profiles are listed too: no dialect has a field for a profile, so
    an id a client cannot see is a preset it cannot select."""
    return {
        "models": [
            {"name": f"models/{served}", "supportedGenerationMethods": list(_METHODS)}
            for served in await profiles.served_ids()
        ]
    }


@router.post("/api/gemini/v1beta/models/{tail:path}", response_model=None)
async def generate(
    tail: str, body: GenerateRequest, engine: EngineDep
) -> JSONResponse | StreamingResponse:
    """The tail is the model name and the method glued by a colon, and the split is at the
    *last* one: `qwen:code:generateContent` selects the profile `code`, and a checkpoint path
    with a colon of its own keeps it."""
    name, _, method = tail.rpartition(":")
    if method not in _METHODS:
        return gemini_error(
            "NOT_FOUND",
            f"{tail!r} names no method this dialect serves: a path here ends in "
            "':generateContent' or ':streamGenerateContent'",
        )

    model_id, profile = await profiles.resolve(name)
    tools = tools_of(body)
    asked = body.generationConfig if body.generationConfig is not None else GenerationConfig()
    if (refusal := refusal_of(asked, tools)) is not None:
        return gemini_error("INVALID_ARGUMENT", refusal)
    # The two levels this dialect can ask for, and never both: a schema is a guarantee here,
    # while the mime type on its own is a demand for JSON measured once the generation is spent.
    strict = asked.responseJsonSchema
    wants_json = asked.responseMimeType == "application/json"
    checked = Checked(None, 1) if strict is None and wants_json else None
    try:
        turns = turns_of(body, None if profile is None else profile.system_prompt)
    except UnreadableImage as unreadable:
        return gemini_error("INVALID_ARGUMENT", str(unreadable))
    if checked is not None:
        turns = (*turns, instruction(checked.schema))
    preset = await profiles.preset(model_id, profile)
    conversation = Chat(turns, tools=tools, reasoning_effort=effort_of(asked, preset))
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        constrained = None if strict is None else await engine.constrain(model_id, strict)
        job = await engine.submit(
            model_id,
            conversation,
            replace(
                generation_options(asked, preset, constrained),
                speculate=await profiles.speculating(model_id, profile),
            ),
        )
    except GrammarRefused as refusal:
        return gemini_error("INVALID_ARGUMENT", str(refusal))
    except NotConstrainable as refusal:
        return gemini_error("INVALID_ARGUMENT", str(refusal))
    except NotQuantizable as refusal:
        return gemini_error("INVALID_ARGUMENT", str(refusal))
    except UnsupportedInput:
        return gemini_error("INVALID_ARGUMENT", unsupported_reason(name, conversation))
    except Exception as failure:
        # A name no checkpoint answers to and one whose load fails are the same answer to the
        # client: this model is not available here, with the reason in the message.
        return gemini_error("NOT_FOUND", f"model {name!r} is not available: {failure}")

    options = dialect_options(tools, asked.maxOutputTokens)
    if method == "streamGenerateContent":
        return StreamingResponse(_frames(job, name, options, checked), media_type=MEDIA_TYPE)

    answer = await collect(job, options)
    if answer.failure is not None:
        return gemini_error("INTERNAL", answer.failure.message)
    text = answer.text
    made = answer.calls
    # A content that called something answered with a call and not with a document: the mime
    # type is about the answer, and this turn has none to check.
    if checked is not None and not made:
        try:
            value = document(text, checked)
        except (MalformedJSON, SchemaViolation) as violation:
            # INTERNAL and not INVALID_ARGUMENT: what did not validate is what this server
            # generated, and of this dialect's four statuses that is the one that does not
            # blame the client for it.
            return gemini_error("INTERNAL", violation_reason(violation, checked))
        # The document and not the text around it: prose or a fence with a value inside it is
        # an answer `json.loads` refuses.
        text = json.dumps(value, ensure_ascii=False)
    # A content that only called something carries no text part: an empty one is a model that
    # answered with nothing, which is not what happened.
    parts: list[dict[str, object]] = [] if made and not text else [{"text": text}]
    parts += [call_part(call) for call in made]
    return JSONResponse(content=reply(name, parts, (answer.reason, answer.usage)))
