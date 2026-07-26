"""The app's HTTP client against a real server — still no engine import in the app."""

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import pytest
import uvicorn
from huggingface_hub import hf_hub_download, snapshot_download

from sideros import GPT2Tokenizer, load_gpt2
from sideros_app import Message, list_models, stream_chat
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
    while not server.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_list_models(base_url: str) -> None:
    assert list_models(base_url) == ["gpt2"]


def test_stream_chat(base_url: str) -> None:
    pieces = list(stream_chat(base_url, "gpt2", [Message("user", "Hello")], max_tokens=8))
    assert len(pieces) > 0
    assert all(isinstance(p, str) for p in pieces)
