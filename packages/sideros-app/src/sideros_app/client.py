"""HTTP client for the sideros server. This package never imports the engine."""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def stream_chat(
    base_url: str,
    model: str,
    messages: list[Message],
    *,
    max_tokens: int = 256,
    timeout: float = 120.0,
) -> Iterator[str]:
    """Yields assistant text pieces from the server's SSE stream."""
    body = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
        "stream": True,
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client(timeout=timeout) as client, client.stream("POST", url, json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                return
            payload = json.loads(data)
            piece = payload["choices"][0]["delta"].get("content")
            if piece:
                yield piece


def list_models(base_url: str, *, timeout: float = 10.0) -> list[str]:
    response = httpx.get(f"{base_url}/api/openai/v1/models", timeout=timeout)
    response.raise_for_status()
    return [entry["id"] for entry in response.json()["data"]]
