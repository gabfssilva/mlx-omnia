"""`/admin/events`: the six resources the app draws, pushed once per change.

A real uvicorn server for the reason the metrics suite has one — `TestClient` runs a
response to completion before handing it over, and this stream only ends when the client
goes away. Every read is bounded by a deadline: a fanout that regressed has to fail the
test that waits for it rather than hang the suite.

The catalog is a real directory the test writes into, because the property that matters
most here is the one nothing in the process raises — a checkpoint appearing on disk has to
reach a client that never asked. FSEvents delivers on its own thread and coalesces on its
own clock, so nothing below asserts on a delay; it asserts on the frame that follows.
"""

import json
import shutil
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from mlx_omnia import CompositeModel, GenerationOptions, Text
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server import catalog, create_app, events
from mlx_omnia.server.engine import Engine
from mlx_omnia.server.store import Store

_DEADLINE_SECONDS = 20.0
"""How long any one frame is waited for. Generous on purpose: FSEvents runs on a clock of
its own, and a suite that fails when the machine is busy is a suite nobody trusts."""

CONFIG: dict[str, object] = {
    "model_type": "qwen3",
    "num_hidden_layers": 2,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "vocab_size": 16,
    "max_position_embeddings": 64,
}


def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
    """This stand answers about state and never generates: every resource under test is a
    fact about the disk, the config or the queue."""
    raise ValueError(f"this stand loads nothing, and nothing should have asked for {model_id!r}")


def _shard(path: Path) -> None:
    """safetensors as the scan reads it: a little-endian u64 with the header's length, the
    JSON header, then the payload its offsets describe."""
    header = json.dumps(
        {"model.norm.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 8)


def _checkpoint(hub: Path, model_id: str) -> None:
    repository = hub / f"models--{model_id.replace('/', '--')}"
    directory = repository / "snapshots" / "head"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(CONFIG))
    _shard(directory / "model.safetensors")
    reference = repository / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("head")


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The hub cache this daemon watches. It is a directory before the app is built: a root
    that does not exist yet is one `observe` does not schedule."""
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(catalog, "HUB_CACHE", hub)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    return hub


@pytest.fixture
def base_url(caches: Path, tmp_path: Path) -> Iterator[str]:
    del caches
    app = create_app(Engine(loader), Store(tmp_path / "server.db"))
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


def frames(
    response: httpx.Response, seconds: float = _DEADLINE_SECONDS
) -> Iterator[tuple[str, object]]:
    """The envelopes, keep-alive comments dropped. One `data:` line per frame is what the
    server writes and what the app's own reader assumes."""
    deadline = time.monotonic() + seconds
    for line in response.iter_lines():
        assert time.monotonic() < deadline, "the stream never said what the test waits for"
        if line.startswith("data: "):
            frame = json.loads(line.removeprefix("data: "))
            yield frame["resource"], frame["value"]


def until(stream: Iterator[tuple[str, object]], resource: str) -> object:
    """The next frame for one resource, ignoring whatever else moved in the meantime."""
    for name, value in stream:
        if name == resource:
            return value
    raise AssertionError(f"the stream ended before a {resource!r} frame")


def listed(value: object) -> list[str]:
    assert isinstance(value, list), f"expected the catalog, got {value!r}"
    return sorted(str(entry["id"]) for entry in value)


def test_the_stream_opens_with_every_resource_once(base_url: str) -> None:
    """A client that connects knows the whole state before anything moves. There is no
    fetch beside the stream any more, so a resource missing from the opening burst is a
    screen that stays blank until that resource happens to change."""
    with httpx.stream("GET", f"{base_url}/admin/events", timeout=30) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        seen = [resource for resource, _ in _take(frames(response), len(events.RESOURCES))]

    assert seen == list(events.RESOURCES)


def _take(stream: Iterator[tuple[str, object]], count: int) -> list[tuple[str, object]]:
    return [next(stream) for _ in range(count)]


def test_a_checkpoint_appearing_on_disk_reaches_a_client_that_never_asked(
    base_url: str, caches: Path
) -> None:
    """What FSEvents is here for. Nothing in this process wrote the checkpoint — an
    `hf download` in another terminal is the real case — so no handler could have announced
    it, and the poll this replaces is the only other way anyone would find out."""
    with httpx.stream("GET", f"{base_url}/admin/events", timeout=30) as response:
        stream = frames(response)
        assert until(stream, "models") == []

        _checkpoint(caches, "Qwen/Qwen3-0.6B")

        assert listed(until(stream, "models")) == ["Qwen/Qwen3-0.6B"]


def test_a_checkpoint_deleted_off_disk_leaves_the_stream_too(
    base_url: str, caches: Path
) -> None:
    """The other direction, which a cache keyed by what it saw last would get wrong: the
    entry has to leave, and the scan behind the frame has to be the one that notices."""
    _checkpoint(caches, "Qwen/Qwen3-0.6B")
    with httpx.stream("GET", f"{base_url}/admin/events", timeout=30) as response:
        stream = frames(response)
        assert listed(until(stream, "models")) == ["Qwen/Qwen3-0.6B"]

        shutil.rmtree(caches / "models--Qwen--Qwen3-0.6B")

        assert until(stream, "models") == []


def test_a_config_patch_comes_back_as_config_and_does_not_resend_the_catalog(
    base_url: str,
) -> None:
    """One resource per frame is what the envelope buys. A client watching the queue must
    not be handed the whole catalog because somebody moved the TTL.

    Nothing is written to disk here, so the frame after the PATCH is the PATCH's: an
    assertion on what comes *next* is only worth making when nothing else could be it."""
    with httpx.stream("GET", f"{base_url}/admin/events", timeout=30) as response:
        stream = frames(response)
        _take(stream, len(events.RESOURCES))

        patched = httpx.patch(
            f"{base_url}/admin/config", json={"idle_ttl_seconds": 123}, timeout=30
        )
        assert patched.status_code == 200, patched.text

        resource, value = next(stream)

    assert resource == "config"
    assert isinstance(value, dict)
    assert value["idle_ttl_seconds"]["value"] == 123


def test_a_burst_of_writes_is_one_frame_and_not_one_per_file(
    base_url: str, caches: Path
) -> None:
    """What keeps the push cheaper than the poll it replaces. A download is thousands of
    FSEvents and one change; a frame per event would rebuild the catalog per chunk, which
    is worse than the two-second poll this exists to remove.

    Counted against a `config` frame rather than against a clock: the PATCH is announced
    with no delay in front of it, so whatever the disk still owed has arrived by the time
    it does."""
    with httpx.stream("GET", f"{base_url}/admin/events", timeout=30) as response:
        stream = frames(response)
        _take(stream, len(events.RESOURCES))

        for index in range(20):
            _checkpoint(caches, f"stand/model-{index}")
        assert len(listed(until(stream, "models"))) == 20

        settled = httpx.patch(
            f"{base_url}/admin/config", json={"idle_ttl_seconds": 321}, timeout=30
        )
        assert settled.status_code == 200, settled.text

        extra = 0
        for resource, _ in stream:
            if resource == "config":
                break
            if resource == "models":
                extra += 1

    assert extra <= 1, f"twenty checkpoints in one burst became {extra + 1} frames"


def _tapped(media_type: str, *chunks: bytes, path: str = "/admin/written") -> FastAPI:
    """A response written in the chunks a test names, so what the tap sees is a socket's
    view and not a whole body handed over at once."""
    app = FastAPI()

    @app.get(path)
    def written() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return StreamingResponse(body(), media_type=media_type)

    app.add_middleware(events.Tap)
    return app


def test_the_tap_prints_every_line_of_a_stream_and_holds_a_frame_split_in_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What a frame is split across is the network's business and never the reader's: a
    tap that printed per chunk would show half a JSON object and call it a line."""
    app = _tapped("text/event-stream", b'data: {"resource"', b': "state"}\n\n: keep-alive\n\n')
    with TestClient(app) as client:
        assert client.get("/admin/written").status_code == 200

    assert capsys.readouterr().out.splitlines() == [
        'sse /admin/written | data: {"resource": "state"}',
        "sse /admin/written | ",
        "sse /admin/written | : keep-alive",
        "sse /admin/written | ",
    ]


def test_a_response_that_is_not_an_event_stream_is_not_tapped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The tap reads the content type and not the path, which is what lets it cover a
    stream nobody named here — and what makes it owe the same silence to a body that is
    not one. A chat completion answered whole is a page of JSON on stdout otherwise."""
    app = _tapped("application/json", b'{"choices": [{"message": {"content": "hello"}}]}')
    with TestClient(app) as client:
        assert client.get("/admin/written").status_code == 200

    assert capsys.readouterr().out == ""


def test_a_generation_is_never_tapped_however_it_is_spelled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dialects are out by prefix. Two reasons, and the second is the one that decides:
    a generation's lines are already on somebody's screen, and the write here is a flush on
    the loop that is carrying those tokens.

    All three dialects, because what excludes them is `/api` and not a list somebody has to
    remember to add a fourth to."""
    frame = b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
    for path in (
        "/api/openai/v1/chat/completions",
        "/api/anthropic/v1/messages",
        "/api/gemini/v1beta/models/qwen",
    ):
        app = _tapped("text/event-stream", frame, path=path)
        with TestClient(app) as client:
            assert client.get(path).status_code == 200

        assert capsys.readouterr().out == "", f"{path} reached stdout"
