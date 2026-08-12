"""Everything Chat asks of the daemon: the session store under `/admin/sessions` and the
OpenAI dialect under `/api/openai/v1`.

The session column holds whatever the client wrote, so it is read rather than cast: a turn
this window cannot draw — a tool result, a message somebody hand-edited into the file — is
dropped instead of rendered as a blank.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Literal, TypedDict, cast
from urllib.parse import quote

from mlx_omnia.app.api import http
from mlx_omnia.app.api.engine import Sample, Speculation

Role = Literal["system", "user", "assistant"]
ROLES = ("system", "user", "assistant")

# `chat/completions`' own vocabulary, which is upstream's. There is no reasoning budget
# beside it because the dialect has no field for one — how long the model may think is set
# on a profile, under Library.
EFFORTS = ["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass
class Metrics:
    """What a turn cost, kept next to the turn it belongs to. Every number is the daemon's
    own, off `x_mlx_omnia` on the usage frame or off the live register — never timed from out
    here, where the load, the queue and the prefill are one wait with no way to tell them
    apart."""

    load_ms: float | None = None
    ttft_ms: float | None = None
    prefill_tokens_per_second: float | None = None
    tokens_per_second: float | None = None
    ceiling_fraction: float | None = None
    speculation: Speculation | None = None
    # Which of the three ended the turn, as the dialect spells it. `length` is the one a
    # reader has to be told about: the answer stops mid-sentence and nothing is wrong.
    finish: str | None = None

    def wire(self) -> dict[str, object]:
        return {
            "load_ms": self.load_ms,
            "ttft_ms": self.ttft_ms,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "tokens_per_second": self.tokens_per_second,
            "ceiling_fraction": self.ceiling_fraction,
            "speculation": self.speculation,
            "finish": self.finish,
        }


@dataclass
class Message:
    role: Role
    content: str
    reasoning: str = ""
    metrics: Metrics | None = None
    error: str | None = None

    def wire(self) -> dict[str, object]:
        body: dict[str, object] = {"role": self.role, "content": self.content}
        if self.reasoning:
            body["reasoning_content"] = self.reasoning
        if self.metrics is not None:
            body["metrics"] = self.metrics.wire()
        if self.error is not None:
            body["error"] = self.error
        return body


@dataclass
class Summary:
    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    message_count: int


@dataclass
class Session:
    id: str
    title: str
    model: str
    created_at: float
    updated_at: float
    messages: list[Message] = field(default_factory=list)

    @property
    def summary(self) -> Summary:
        return Summary(
            self.id,
            self.title,
            self.model,
            self.created_at,
            self.updated_at,
            len(self.messages),
        )


def _number(value: object) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _speculation(value: object) -> Speculation | None:
    if not isinstance(value, dict):
        return None
    trio = [_number(value.get(key)) for key in ("rounds", "proposed", "accepted")]
    if any(one is None for one in trio):
        return None
    rounds, proposed, accepted = trio
    return {
        "rounds": int(rounds or 0),
        "proposed": int(proposed or 0),
        "accepted": int(accepted or 0),
    }


def _metrics(value: object) -> Metrics | None:
    if not isinstance(value, dict):
        return None
    return Metrics(
        load_ms=_number(value.get("load_ms")),
        ttft_ms=_number(value.get("ttft_ms")),
        prefill_tokens_per_second=_number(value.get("prefill_tokens_per_second")),
        tokens_per_second=_number(value.get("tokens_per_second")),
        ceiling_fraction=_number(value.get("ceiling_fraction")),
        speculation=_speculation(value.get("speculation")),
        finish=value.get("finish") if isinstance(value.get("finish"), str) else None,
    )


def _message(value: object) -> Message | None:
    if not isinstance(value, dict):
        return None
    role, content = value.get("role"), value.get("content")
    if role not in ROLES or not isinstance(content, str):
        return None
    reasoning = value.get("reasoning_content")
    error = value.get("error")
    return Message(
        role=cast(Role, role),
        content=content,
        reasoning=reasoning if isinstance(reasoning, str) else "",
        metrics=_metrics(value.get("metrics")),
        error=error if isinstance(error, str) and error else None,
    )


def _session(raw: object) -> Session:
    body = cast(dict[str, object], raw)
    listed = body.get("messages")
    return Session(
        id=str(body["id"]),
        title=str(body["title"]),
        model=str(body["model"]),
        created_at=float(cast(float, body["created_at"])),
        updated_at=float(cast(float, body["updated_at"])),
        messages=[
            turn
            for turn in (_message(one) for one in (listed if isinstance(listed, list) else []))
            if turn is not None
        ],
    )


def _at(identifier: str) -> str:
    return f"/admin/sessions/{quote(identifier, safe='')}"


async def list_sessions() -> list[Summary]:
    body = cast(dict[str, list[dict[str, object]]], await http.get("/admin/sessions"))
    return [
        Summary(
            id=str(row["id"]),
            title=str(row["title"]),
            model=str(row["model"]),
            created_at=float(cast(float, row["created_at"])),
            updated_at=float(cast(float, row["updated_at"])),
            message_count=int(cast(int, row["message_count"])),
        )
        for row in body["sessions"]
    ]


async def create_session(title: str | None = None, model: str = "") -> Session:
    body: dict[str, object] = {"model": model}
    if title is not None:
        body["title"] = title
    return _session(await http.send("POST", "/admin/sessions", body))


async def get_session(identifier: str) -> Session:
    return _session(await http.get(_at(identifier)))


async def patch_session(
    identifier: str, title: str | None = None, model: str | None = None
) -> Session:
    body: dict[str, object] = {}
    if title is not None:
        body["title"] = title
    if model is not None:
        body["model"] = model
    return _session(await http.send("PATCH", _at(identifier), body))


async def put_messages(identifier: str, messages: list[Message]) -> Session:
    return _session(
        await http.send(
            "PUT", f"{_at(identifier)}/messages", {"messages": [m.wire() for m in messages]}
        )
    )


async def delete_session(identifier: str) -> None:
    await http.send("DELETE", _at(identifier))


async def served_models() -> list[str]:
    body = cast(dict[str, list[dict[str, str]]], await http.get("/api/openai/v1/models"))
    return [entry["id"] for entry in body["data"]]


# ── the knobs a request carries ───────────────────────────────────────────


@dataclass(frozen=True)
class Params:
    temperature: float = 1.0
    top_p: float = 1.0
    # None is the dialect's "no cut": the field is left out of the body.
    top_k: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096
    # `default` leaves the field out, which is the checkpoint's own template deciding.
    reasoning_effort: str = "default"
    system_prompt: str = ""


DEFAULTS = Params()


def params_of(entry: object, base: Params = DEFAULTS) -> Params:
    """The knobs a chat opens on: the checkpoint's declared defaults over the dialect's,
    which is the same order the daemon resolves them in.

    `max_tokens`, the effort and the system prompt are not in that file and stay as they
    were: they are the reader's, not the checkpoint's.
    """
    if not isinstance(entry, dict):
        return base
    declared = entry.get("defaults")
    if not isinstance(declared, dict):
        return base
    def knob(name: str, fallback: float) -> float:
        # Not `or`: a declared 0.0 is falsy and is the one value that matters most here.
        # `do_sample: false` reaches this file as `temperature: 0.0` — the checkpoint saying
        # greedy — and falling back to the dialect's 1.0 samples where it said not to, which
        # also costs the turn its speculation (that is greedy-only).
        value = _number(declared.get(name))
        return fallback if value is None else value

    return replace(
        base,
        temperature=knob("temperature", DEFAULTS.temperature),
        top_p=knob("top_p", DEFAULTS.top_p),
        top_k=int(top_k) if (top_k := _number(declared.get("top_k"))) is not None else None,
        min_p=knob("min_p", DEFAULTS.min_p),
        repetition_penalty=knob("repetition_penalty", DEFAULTS.repetition_penalty),
    )


def _body(model: str, turns: list[dict[str, str]], params: Params) -> dict[str, object]:
    asked: dict[str, object] = {
        "model": model,
        "messages": turns,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
    }
    if params.top_k is not None:
        asked["top_k"] = params.top_k
    if params.min_p > 0:
        asked["min_p"] = params.min_p
    if params.repetition_penalty > 1:
        asked["repetition_penalty"] = params.repetition_penalty
    if params.reasoning_effort != "default":
        asked["reasoning_effort"] = params.reasoning_effort
    return asked


class Timings(TypedDict):
    """app.py `_timings`: what the turn cost in seconds, on the usage frame under a key
    the dialect does not define."""

    load_seconds: float | None
    ttft_seconds: float | None
    prefill_tokens_per_second: float | None
    tokens_per_second: float | None
    bytes_per_token: float | None
    ceiling_fraction: float | None
    speculation: Speculation | None


@dataclass
class Frame:
    kind: Literal["content", "reasoning", "usage", "timings", "finish", "error"]
    text: str = ""
    total_tokens: int = 0
    timings: Timings | None = None


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000


def metrics_of(timings: Timings, finish: str | None) -> Metrics:
    """The turn's numbers as they are kept, from the frame that closes it."""
    return Metrics(
        load_ms=_ms(timings["load_seconds"]),
        ttft_ms=_ms(timings["ttft_seconds"]),
        prefill_tokens_per_second=timings["prefill_tokens_per_second"],
        tokens_per_second=timings["tokens_per_second"],
        ceiling_fraction=timings["ceiling_fraction"],
        speculation=timings["speculation"],
        finish=finish,
    )


def metrics_of_sample(sample: Sample) -> Metrics:
    """The same numbers while the turn is still being written, off the register's live
    entry. `finish` is nobody's yet: the turn has not ended."""
    return Metrics(
        load_ms=_ms(sample["load_seconds"]),
        ttft_ms=_ms(sample["ttft"]),
        prefill_tokens_per_second=sample["prefill_tokens_per_second"],
        tokens_per_second=sample["tokens_per_second"],
        ceiling_fraction=sample["ceiling_fraction"],
        speculation=sample["speculation"],
    )


async def completion(
    model: str, turns: list[dict[str, str]], params: Params
) -> AsyncGenerator[Frame]:
    """The frames of one turn, in order. `[DONE]` ends it; an `error` frame is yielded and
    then ends it, because that is the whole failure — the stream will not say more."""
    async for data in http.post_events(
        "/api/openai/v1/chat/completions", _body(model, turns, params)
    ):
        if data == "[DONE]":
            return
        chunk = cast(dict[str, object], json.loads(data))
        error = chunk.get("error")
        if isinstance(error, dict):
            yield Frame("error", text=str(error.get("message", "the request failed")))
            return
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = cast(dict[str, object], choices[0])
            delta = cast(dict[str, object], choice.get("delta") or {})
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            if isinstance(reasoning, str) and reasoning:
                yield Frame("reasoning", text=reasoning)
            if isinstance(content, str) and content:
                yield Frame("content", text=content)
            finish = choice.get("finish_reason")
            if isinstance(finish, str) and finish:
                yield Frame("finish", text=finish)
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            yield Frame("usage", total_tokens=int(cast(int, usage["total_tokens"])))
        extra = chunk.get("x_mlx_omnia")
        if isinstance(extra, dict):
            yield Frame("timings", timings=cast(Timings, extra))
