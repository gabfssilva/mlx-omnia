"""One error envelope per dialect, on the wire shape its own SDK reads.

Each envelope is built apart from the response because a stream carries it too: every one of
these SDKs raises on a frame whose JSON has an `error`, and that is the only way to fail a
generation that already answered 200.
"""

from __future__ import annotations

from typing import Literal

from fastapi.responses import JSONResponse

type GeminiStatus = Literal["INVALID_ARGUMENT", "NOT_FOUND", "INTERNAL", "UNAUTHENTICATED"]

_GEMINI_CODES: dict[GeminiStatus, int] = {
    "INVALID_ARGUMENT": 400,
    "NOT_FOUND": 404,
    "INTERNAL": 500,
    "UNAUTHENTICATED": 401,
}


def openai_envelope(
    message: str, code: str, kind: str = "invalid_request_error"
) -> dict[str, str]:
    return {"message": message, "type": kind, "code": code}


def openai_error(
    status: int, message: str, code: str, kind: str = "invalid_request_error"
) -> JSONResponse:
    """The envelope of `chat/completions` and `/v1/responses` alike."""
    return JSONResponse(
        status_code=status, content={"error": openai_envelope(message, code, kind)}
    )


def anthropic_envelope(kind: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def anthropic_error(status: int, kind: str, message: str) -> JSONResponse:
    """`kind` is Anthropic's own vocabulary — `invalid_request_error`, `not_found_error`,
    `api_error` — and travels beside the status rather than instead of it."""
    return JSONResponse(status_code=status, content=anthropic_envelope(kind, message))


def gemini_envelope(status: GeminiStatus, message: str) -> dict[str, object]:
    return {
        "error": {"code": _GEMINI_CODES[status], "message": message, "status": status}
    }


def gemini_error(status: GeminiStatus, message: str) -> JSONResponse:
    """The status name is the code's canonical spelling, so the two cannot disagree."""
    return JSONResponse(
        status_code=_GEMINI_CODES[status], content=gemini_envelope(status, message)
    )
