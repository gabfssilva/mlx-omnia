"""Gate P5: official OpenAI SDK against a real HTTP server process."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import httpx
import mlx.core as mx
import pytest
import uvicorn
from huggingface_hub import hf_hub_download, snapshot_download
from openai import NotFoundError, OpenAI

from sideros import GPT2Tokenizer, load_gpt2
from sideros_server import Engine, create_app


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    directory = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
    model = load_gpt2(directory, dtype=mx.float16)
    tokenizer = GPT2Tokenizer.from_files(
        Path(hf_hub_download("gpt2", "vocab.json")),
        Path(hf_hub_download("gpt2", "merges.txt")),
    )
    app = create_app(Engine(model, tokenizer, "gpt2"))

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def client(base_url: str) -> OpenAI:
    return OpenAI(base_url=f"{base_url}/api/openai/v1", api_key="unused")


def ask(client: OpenAI, prompt: str, max_tokens: int = 16) -> str:
    response = client.chat.completions.create(
        model="gpt2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    assert content is not None
    return content


def test_health(base_url: str) -> None:
    payload = httpx.get(f"{base_url}/admin/health").json()
    assert payload == {"status": "ok", "model": "gpt2"}


def test_models_list(client: OpenAI) -> None:
    ids = [m.id for m in client.models.list()]
    assert ids == ["gpt2"]


def test_unknown_model_is_openai_error(client: OpenAI) -> None:
    with pytest.raises(NotFoundError):
        client.chat.completions.create(
            model="nope", messages=[{"role": "user", "content": "x"}]
        )


def test_chat_completion_deterministic(client: OpenAI) -> None:
    first = ask(client, "The capital of France is")
    second = ask(client, "The capital of France is")
    assert first == second
    assert len(first) > 0


def test_streaming_matches_non_streaming(client: OpenAI) -> None:
    prompt = "Once upon a time"
    stream = client.chat.completions.create(
        model="gpt2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        stream=True,
    )
    pieces: list[str] = []
    finish = None
    for chunk in stream:
        finish = chunk.choices[0].finish_reason or finish
        pieces.append(chunk.choices[0].delta.content or "")
    assert finish == "stop"
    assert "".join(pieces) == ask(client, prompt)


def test_concurrent_requests_serialize(client: OpenAI) -> None:
    prompts = ["My favorite color is", "The tallest mountain is"]
    serial = [ask(client, p) for p in prompts]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(partial(ask, client), prompts))
    assert concurrent == serial


def test_empty_messages_is_rejected_and_worker_survives(base_url: str, client: OpenAI) -> None:
    body = {"model": "gpt2", "messages": [], "max_tokens": 4}
    response = httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert len(ask(client, "Hi", max_tokens=4)) > 0


def test_lone_surrogate_is_reported_and_worker_survives(base_url: str, client: OpenAI) -> None:
    body = {
        "model": "gpt2",
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
        "model": "gpt2",
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
