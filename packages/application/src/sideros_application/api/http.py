"""How this window reaches the daemon, and the two shapes an answer arrives in: a body,
or a stream of them.

The house refusal is FastAPI's `{"detail": ...}`, which is what a screen shows rather than
a status code.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import cast
from urllib.parse import quote

import httpx

BASE = os.environ.get("SIDEROS_DAEMON_URL", "http://127.0.0.1:8642").rstrip("/")


class Refused(Exception):
    pass


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BASE, timeout=30)


def at(identifier: str) -> str:
    """A model id in a path segment. The route is a `:path`, so the slash in `org/name`
    stays a slash and only the rest is escaped."""
    return "/".join(quote(part, safe="") for part in identifier.split("/"))


async def get(path: str) -> object:
    async with client() as http:
        answer = await http.get(path)
        answer.raise_for_status()
        return answer.json()


async def send(method: str, path: str, body: object | None = None) -> object:
    async with client() as http:
        answer = await http.request(
            method, path, json=body, timeout=None if method == "POST" else 30
        )
        if answer.is_error:
            raise Refused(_detail(answer))
        return None if answer.status_code == 204 else answer.json()


async def events(http: httpx.AsyncClient, path: str) -> AsyncGenerator[dict[str, object]]:
    """text/event-stream carrying whole JSON states on `data:` lines, plus `: keep-alive`
    comments that mean nothing but the connection is warm."""
    async with http.stream("GET", path) as answer:
        answer.raise_for_status()
        held: list[str] = []
        async for line in answer.aiter_lines():
            if line.startswith("data:"):
                held.append(line[5:].lstrip())
            elif line == "" and held:
                yield cast(dict[str, object], json.loads("\n".join(held)))
                held = []


async def post_events(path: str, body: object) -> AsyncGenerator[str]:
    """The `data:` payloads of a stream a POST opens, verbatim — the dialect's own frames
    are not this module's to read, and `[DONE]` is one of them."""
    async with (
        httpx.AsyncClient(base_url=BASE, timeout=None) as http,
        http.stream("POST", path, json=body) as answer,
    ):
        if answer.is_error:
            await answer.aread()
            raise Refused(_detail(answer))
        held: list[str] = []
        async for line in answer.aiter_lines():
            if line.startswith("data:"):
                held.append(line[5:].lstrip())
            elif line == "" and held:
                yield "\n".join(held)
                held = []


def _detail(answer: httpx.Response) -> str:
    try:
        body = answer.json()
    except ValueError:
        return answer.reason_phrase
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
    return answer.reason_phrase
