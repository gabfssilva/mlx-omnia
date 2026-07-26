import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from sideros_server.engine import Engine, Job

_KEEP_ALIVE_SECONDS = 0.5


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    stream: bool = False


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


def _error(
    status: int, message: str, code: str, kind: str = "invalid_request_error"
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": code}},
    )


def _chunk(
    request_id: str, created: int, model: str, delta: dict[str, str], finish: str | None
) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _drain(job: Job) -> list[str]:
    pieces: list[str] = []
    while (piece := await job.chunks.get()) is not None:
        pieces.append(piece)
    return pieces


async def _events(job: Job, request_id: str, created: int, model: str) -> AsyncIterator[str]:
    try:
        yield _chunk(request_id, created, model, {"role": "assistant", "content": ""}, None)
        while True:
            try:
                piece = await asyncio.wait_for(job.chunks.get(), _KEEP_ALIVE_SECONDS)
            except TimeoutError:
                # Keeps the connection warm through a long prefill.
                yield ": keep-alive\n\n"
                continue
            if piece is None:
                break
            yield _chunk(request_id, created, model, {"content": piece}, None)
        yield _chunk(request_id, created, model, {}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        job.cancel()


@router.get("/admin/health")
async def health(engine: EngineDep) -> dict[str, str]:
    return {"status": "ok", "model": engine.model_id}


@router.get("/api/openai/v1/models")
async def models(engine: EngineDep) -> dict[str, object]:
    return {
        "object": "list",
        "data": [{"id": engine.model_id, "object": "model", "created": 0, "owned_by": "sideros"}],
    }


@router.post("/api/openai/v1/chat/completions", response_model=None)
async def chat(request: ChatRequest, engine: EngineDep) -> JSONResponse | StreamingResponse:
    if request.model != engine.model_id:
        return _error(404, f"model {request.model!r} not found", "model_not_found")

    prompt = "\n\n".join(message.content for message in request.messages)
    if not prompt:
        return _error(400, "messages must contain non-empty content", "empty_prompt")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    job = await engine.submit(prompt, request.max_tokens)

    if request.stream:
        return StreamingResponse(
            _events(job, request_id, created, engine.model_id),
            media_type="text/event-stream",
        )

    try:
        pieces = await _drain(job)
    finally:
        job.cancel()
    if job.error is not None:
        return _error(500, job.error, "generation_failed", kind="server_error")
    return JSONResponse(
        content={
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(pieces)},
                    "finish_reason": "stop",
                }
            ],
        }
    )


def create_app(engine: Engine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.include_router(router)
    return app
