"""The api key: no key configured is an open daemon, which is the 127.0.0.1 default.

Once one is set it covers every route but `/admin/health` — that is how the app finds the
daemon, and a discovery probe answering 401 is a daemon the app draws as down.

The key is read from the database **per request** rather than at boot, so a rotation — and a
revocation, which is the one that matters — starts counting on the next call. That is what
makes `api_key` an `applied` field and not a `restart` one.

The refusal is encoded in the dialect of whoever asked, chosen by path prefix: the SDKs read
the error out of the body, and Gemini's prints `None None.` when it gets somebody else's
envelope. `/admin` and anything unrouted get OpenAI's.
"""

import hmac

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from mlx_omnia.server.api.errors import anthropic_error, gemini_error, openai_error
from mlx_omnia.server.services import config

_EXEMPT = frozenset({"/admin/health"})

_MESSAGE = (
    "api key missing or wrong: send it as `Authorization: Bearer <key>`, `x-api-key` or "
    "`x-goog-api-key`"
)


def _refusal(path: str) -> JSONResponse:
    if path.startswith("/api/anthropic/"):
        return anthropic_error(401, "authentication_error", _MESSAGE)
    if path.startswith("/api/gemini/"):
        return gemini_error("UNAUTHENTICATED", _MESSAGE)
    return openai_error(401, _MESSAGE, "invalid_api_key")


def _presented(headers: Headers) -> str | None:
    """The three carriers in circulation, one per dialect: OpenAI signs with `Authorization`,
    Anthropic with `x-api-key`, Gemini with `x-goog-api-key`. All three are accepted whatever
    the route, because the header is the client's habit and not the endpoint's."""
    authorization = headers.get("authorization")
    if authorization is not None and authorization.startswith("Bearer "):
        return authorization.removeprefix("Bearer ")
    return headers.get("x-api-key") or headers.get("x-goog-api-key")


def _matches(presented: str, key: str) -> bool:
    """Two encodings and not one: what arrived was decoded latin-1, which is the wire's own,
    so latin-1 gives back the bytes the client sent; the stored key is text, and its bytes
    are utf-8. Comparing the two `str`s instead is what `compare_digest` refuses outright the
    moment either of them has a non-ascii character in it."""
    return hmac.compare_digest(presented.encode("latin-1"), key.encode())


class ApiKey:
    """Pure ASGI rather than `BaseHTTPMiddleware`: the token routes are SSE, and this way
    nothing sits between the generator and the socket."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _EXEMPT:
            await self._app(scope, receive, send)
            return
        key = (await config.current()).api_key
        if key is None:
            await self._app(scope, receive, send)
            return
        presented = _presented(Headers(scope=scope))
        if presented is not None and _matches(presented, key):
            await self._app(scope, receive, send)
            return
        await _refusal(scope["path"])(scope, receive, send)


def check_bind(host: str, api_key: str | None) -> None:
    """Refuses to come up rather than serving open: off the loopback the key is the only
    thing between the weights and the network.

    A host that is not an address — `localhost` aside — counts as off the loopback: it
    resolves to whatever the resolver says it does, and the conservative reading of that is
    the one that asks for a key.
    """
    if config.loopback(host) or api_key is not None:
        return
    raise SystemExit(
        f"refusing to listen on {host} without an api key: set one with PATCH /admin/config,"
        " or bind 127.0.0.1"
    )
