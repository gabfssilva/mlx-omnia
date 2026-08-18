"""`/api/anthropic/v1/*` — the routes themselves."""

import asyncio
import uuid
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omnia import Chat, UnsupportedInput
from mlx_omnia.engine.chat import template_of
from mlx_omnia.engine.grammar import GrammarRefused
from mlx_omnia.engine.language import tokenizer_of
from mlx_omnia.server.api import sse
from mlx_omnia.server.api.anthropic.codec import (
    declared_tools,
    encode_error,
    generation_options,
    reasoning_shown,
    to_conversation,
)
from mlx_omnia.server.api.anthropic.encode import consume_options, encode_answer
from mlx_omnia.server.api.anthropic.models import CountRequest, Message, MessagesRequest, TextBlock
from mlx_omnia.server.api.anthropic.stream import encode_stream
from mlx_omnia.server.api.responses import UnreadableImage, unsupported_reason
from mlx_omnia.server.generation.collect import collect
from mlx_omnia.server.generation.consume import consume
from mlx_omnia.server.runtime.engine import Engine, NotConstrainable, NotQuantizable
from mlx_omnia.server.services import profiles


async def engine_of(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(engine_of)]

router = APIRouter()


@router.get("/api/anthropic/v1/models")
async def models() -> dict[str, object]:
    """The catalog, and every name each model answers to. `has_more` is false rather than
    absent: the SDK pages until something says to stop."""
    return {
        "data": [
            {
                "id": served,
                "type": "model",
                "display_name": served,
                "created_at": "1970-01-01T00:00:00Z",
            }
            for served in await profiles.served_ids()
        ],
        "has_more": False,
    }


@router.get("/api/anthropic/v1/models/{model_id:path}")
async def model(model_id: str) -> JSONResponse:
    """One entry of the listing above, by the name it answers to. `{model_id:path}` because a
    served id carries slashes. A name nothing serves is `not_found_error` and not an empty
    object: the SDK reads this to decide whether a model exists."""
    if model_id not in await profiles.served_ids():
        return encode_error(404, "not_found_error", f"model {model_id!r} is not available here")
    return JSONResponse(
        content={
            "id": model_id,
            "type": "model",
            "display_name": model_id,
            "created_at": "1970-01-01T00:00:00Z",
        }
    )


def _pictured(conversation: Chat) -> bool:
    """Whether an image reached the turns, in the shape `content_of` leaves them."""
    return any(
        not isinstance(content, str) and any(part["type"] == "image" for part in content)
        for content in (message["content"] for message in conversation.messages)
    )


@router.post("/api/anthropic/v1/messages/count_tokens", response_model=None)
async def count_tokens(request: CountRequest, engine: EngineDep) -> JSONResponse:
    """What this conversation costs to send, in the tokens the checkpoint itself counts.

    Only a resident model answers: the caller is a client counting a context window as it
    fills, and a load is seconds and tens of gigabytes asked for on purpose. The count is of
    the rendered prompt and not of the turns — what the tokens are is decided by the chat
    template — and an image is refused instead of skipped, because how many tokens a picture
    becomes is the checkpoint's own processor's to say.
    """
    model_id, profile = await profiles.resolve(request.model)
    if model_id not in engine.resident:
        return encode_error(
            409,
            "invalid_request_error",
            f"{model_id!r} is not resident: load it before counting tokens with it",
        )
    # The body before the checkpoint: what the client sent is the client's to fix, and hearing
    # about the deployment first sends it looking at the wrong end of the request.
    try:
        conversation = to_conversation(
            request,
            None if profile is None else profile.system_prompt,
            None if profile is None else profile.sampling.reasoning_effort,
        )
    except UnreadableImage as unreadable:
        return encode_error(400, "invalid_request_error", str(unreadable))
    if _pictured(conversation):
        return encode_error(
            400,
            "invalid_request_error",
            "an image's tokens are counted by the checkpoint's own processor and not by this "
            "route, which renders the conversation as text: count the turns without it",
        )
    resolved = await engine.resolve(model_id)
    template = template_of(resolved)
    tokenizer = tokenizer_of(resolved)
    if template is None:
        return encode_error(
            400, "invalid_request_error", unsupported_reason(request.model, Chat(()))
        )
    if tokenizer is None:
        return encode_error(400, "invalid_request_error", f"{model_id!r} exposes no tokenizer")
    try:
        prompt = template.render(conversation)
    except Exception as refusal:
        # A template that will not render these turns is a prompt this conversation cannot
        # become, and the client is the only one who can change it.
        return encode_error(400, "invalid_request_error", str(refusal))

    # Read out in the thread, not after it: `encode` hands the ids over as it makes them.
    def encode() -> list[int]:
        return list(tokenizer.encode(prompt))

    ids = await asyncio.to_thread(encode)
    return JSONResponse(content={"input_tokens": len(ids)})


def _said(message: Message) -> bool:
    """Whether the client wrote anything in this message. A turn that only carries the result
    of a call said what it had to say."""
    content = message.content
    if isinstance(content, str):
        return bool(content)
    return any(block.text if isinstance(block, TextBlock) else True for block in content)


@router.post("/api/anthropic/v1/messages", response_model=None)
async def messages(request: MessagesRequest, engine: EngineDep) -> JSONResponse | StreamingResponse:
    if not any(_said(message) for message in request.messages):
        return encode_error(400, "invalid_request_error", "messages must contain non-empty content")
    config = request.output_config
    schema = None if config is None or config.format is None else config.format.definition
    if schema is not None and declared_tools(request):
        return encode_error(
            400,
            "invalid_request_error",
            "output_config.format and tools cannot both be honoured: the grammar constrains "
            "decoding to the schema from the first token, so the model cannot write a call "
            "however it is offered. Send the schema alone, or offer no tools.",
        )

    model_id, profile = await profiles.resolve(request.model)
    preset = await profiles.preset(model_id, profile)
    try:
        conversation = to_conversation(
            request,
            None if profile is None else profile.system_prompt,
            preset.reasoning_effort,
        )
    except UnreadableImage as unreadable:
        return encode_error(400, "invalid_request_error", str(unreadable))
    message_id = f"msg_{uuid.uuid4().hex}"
    speculate = await profiles.speculating(model_id, profile)
    try:
        # One walk per generation and never one shared between two: the grammar behind it is
        # the engine's to keep, the walk is this request's.
        walk = None if schema is None else await engine.constrain(model_id, schema)
        job = await engine.submit(
            model_id,
            conversation,
            replace(generation_options(request, preset, walk), speculate=speculate),
        )
    except GrammarRefused as refusal:
        # The compiler's own words — `Unimplemented keys: ["uniqueItems"]` is a reason where
        # "grammar error" is not, and it is what tells the client which keyword to drop.
        return encode_error(400, "invalid_request_error", str(refusal))
    except NotConstrainable as refusal:
        return encode_error(400, "invalid_request_error", str(refusal))
    except NotQuantizable as refusal:
        return encode_error(400, "invalid_request_error", str(refusal))
    except UnsupportedInput:
        return encode_error(
            400, "invalid_request_error", unsupported_reason(request.model, conversation)
        )
    except Exception as error:
        return encode_error(
            404, "not_found_error", f"model {request.model!r} is not available: {error}"
        )

    options = consume_options(request, bool(conversation.tools))
    shown = reasoning_shown(request)
    if request.stream:
        return StreamingResponse(
            encode_stream(consume(job, options), message_id, request.model, shown, job.meter),
            media_type=sse.MEDIA_TYPE,
        )

    completion = await collect(job, options)
    if completion.failure is not None:
        return encode_error(500, "api_error", completion.failure.message)
    return JSONResponse(content=encode_answer(completion, message_id, request.model, shown))
