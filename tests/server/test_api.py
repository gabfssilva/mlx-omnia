"""Gate P5: official OpenAI SDK against a real HTTP server process."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib.metadata import version
from pathlib import Path

import httpx
import pytest
from huggingface_hub import snapshot_download
from openai import NotFoundError, OpenAI

from mlx_omnia import ByteLevelBPE, Chat, chat_template
from tests.server import openai_stand
from tests.server.openai_script import CALLER, MODEL
from tests.server.openai_stand import (
    _health,
    ask,
    resident,
)

fresh_state = openai_stand.fresh_state
"""Overrides `conftest`'s per-test wipe, which would delete the database under the
module server while it is still answering."""

pytest_plugins = ("tests.server.openai_stand",)
"""The per-file server. `fresh_state` is imported to override `conftest`'s per-test wipe,
which would delete the database under a server that is still answering."""


def test_health_says_who_is_answering_and_for_how_long(base_url: str) -> None:
    """The one route a key does not cover, and the only place a client that did not start the
    daemon can read the process it is talking to: the app draws it, and the CLI decides
    between talking to one and starting one."""
    payload = _health(base_url)
    assert payload["status"] == "ok"
    assert payload["pid"] == os.getpid(), "the stand runs the server in this process"
    uptime = payload["uptime"]
    assert isinstance(uptime, float) and uptime > 0
    assert payload["version"] == version("mlx_omnia")


def test_the_admin_routes_are_mounted_on_the_real_app(base_url: str) -> None:
    """Each `/admin` group lives in its own module and is wired in `create_app`. Their own
    suites mount the routers on a throwaway FastAPI, so nothing there would notice a group
    that never reaches the server the daemon actually serves."""
    for path in ("system", "models", "jobs", "state", "metrics", "config", "benchmarks/runs"):
        assert httpx.get(f"{base_url}/admin/{path}", timeout=60).status_code == 200, path
    # A body the route refuses, not a download: what is being asked is whether the route
    # is there at all, and 404 is the answer that says it is not.
    for path in ("models", "quantizations", "models/nothing/tokenize"):
        assert httpx.post(f"{base_url}/admin/{path}", json={}, timeout=30).status_code != 404, path
    # Residency answers 404 for an id nobody loaded, which is also what an unmounted route
    # answers: the two are told apart by whose 404 it is.
    refused = httpx.delete(f"{base_url}/admin/models/nothing/residency", timeout=30)
    assert refused.status_code == 404 and "not resident" in refused.json()["detail"]
    # Each dialect is a module of its own too, and the base URL the Server screen tells the
    # user to copy is only true if its router reached the app the daemon serves.
    for path in ("anthropic/v1/models", "gemini/v1beta/models"):
        assert httpx.get(f"{base_url}/api/{path}", timeout=30).status_code == 200, path
    dialects = (
        "openai/v1/responses",
        "anthropic/v1/messages",
        "gemini/v1beta/models/x:generateContent",
    )
    for path in dialects:
        assert httpx.post(f"{base_url}/api/{path}", json={}, timeout=30).status_code != 404, path


def test_nothing_is_resident_until_a_request_names_a_model(base_url: str, client: OpenAI) -> None:
    """The dialect lists the catalog, so `models.list()` answers the same before and after
    a load — what proves the load was lazy is the resident set, and residency is an
    `/admin` question because no dialect's schema carries the notion.

    The checkpoint is on disk before the server starts (see the fixture), which is what
    makes the first assertion a statement about the catalog rather than about whatever a
    previous run happened to leave cached.
    """
    assert MODEL in [m.id for m in client.models.list()]
    assert _health(base_url)["models"] == []
    assert resident(base_url) == []

    ask(client, "Hi", max_tokens=4)

    assert MODEL in [m.id for m in client.models.list()]
    assert _health(base_url)["models"] == [MODEL]
    assert resident(base_url) == [MODEL]


def test_unknown_model_is_openai_error(client: OpenAI) -> None:
    with pytest.raises(NotFoundError):
        client.chat.completions.create(model="nope", messages=[{"role": "user", "content": "x"}])


def test_chat_completion_deterministic(client: OpenAI) -> None:
    first = ask(client, "The capital of France is")
    second = ask(client, "The capital of France is")
    assert first == second
    assert len(first) > 0


def test_streaming_matches_non_streaming(client: OpenAI) -> None:
    prompt = "Once upon a time"
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0,
        stream=True,
    )
    pieces: list[str] = []
    finish = None
    for chunk in stream:
        finish = chunk.choices[0].finish_reason or finish
        pieces.append(chunk.choices[0].delta.content or "")
    whole = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=16, temperature=0
    )
    # The budget cuts a 0.6B mid-sentence at sixteen tokens, and both shapes have to say so:
    # what this test is about is the two agreeing, the reason included.
    assert finish == whole.choices[0].finish_reason == "length"
    assert "".join(pieces) == whole.choices[0].message.content


def test_concurrent_requests_serialize(client: OpenAI) -> None:
    prompts = ["My favorite color is", "The tallest mountain is"]
    serial = [ask(client, p) for p in prompts]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(partial(ask, client), prompts))
    assert concurrent == serial


def test_empty_messages_is_rejected_and_worker_survives(base_url: str, client: OpenAI) -> None:
    body = {"model": MODEL, "messages": [], "max_tokens": 4}
    response = httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert len(ask(client, "Hi", max_tokens=4)) > 0


def test_lone_surrogate_is_reported_and_worker_survives(base_url: str, client: OpenAI) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "\ud800"}],
        "max_tokens": 4,
    }
    response = httpx.post(
        f"{base_url}/api/openai/v1/chat/completions",
        content=json.dumps(body, ensure_ascii=True).encode("ascii"),
        headers={"content-type": "application/json"},
        timeout=30,
    )
    assert response.status_code >= 400
    assert len(ask(client, "Hi", max_tokens=4)) > 0


def test_client_cancel_stops_generation(base_url: str, client: OpenAI) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a very long story."}],
        "max_tokens": 512,
        "stream": True,
    }
    with (
        httpx.Client() as http,
        http.stream("POST", f"{base_url}/api/openai/v1/chat/completions", json=body) as r,
    ):
        for i, _line in enumerate(r.iter_lines()):
            if i >= 3:
                break
    # If the 512-token job kept running, this small request would wait ~1s behind it.
    start = time.perf_counter()
    ask(client, "Hi", max_tokens=4)
    assert time.perf_counter() - start < 0.5


def test_usage_counts_the_rendered_prompt_and_the_ids_emitted(client: OpenAI) -> None:
    """`prompt_tokens` is not the message the client sent: what reaches the model is the
    conversation the checkpoint's own template renders, and that text exists nowhere but
    inside the engine — the number has to come out of the generation itself."""
    prompt = "The capital of France is"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0,
    )
    usage = response.usage
    assert usage is not None

    directory = Path(snapshot_download(MODEL))
    template = chat_template(directory)
    assert template is not None
    rendered = template.render(Chat(({"role": "user", "content": prompt},)))
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    assert usage.prompt_tokens == len(list(tokenizer.encode(rendered)))
    assert 0 < usage.completion_tokens <= 16
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.prompt_tokens_details is not None, "the field is written even at zero"


def test_a_turn_with_nothing_stored_says_zero_instead_of_saying_nothing(client: OpenAI) -> None:
    """Absent reads as a server that does not carry the field and zero as a miss, and a
    client tuning a conversation to hit the cache needs to tell them apart. The scripted
    model never reuses: it fills its own meter and there is no trie under it."""
    response = client.chat.completions.create(
        model=CALLER, messages=[{"role": "user", "content": "Weather in Paris?"}], max_tokens=32
    )

    usage = response.usage
    assert usage is not None and usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 0
