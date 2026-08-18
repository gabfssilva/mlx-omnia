"""The stand the OpenAI-dialect suite runs against: one real HTTP server process.

The server is a module fixture, so each file of the suite gets its own daemon over its own
state directory — the wipe is module-scoped here for that reason, overriding the per-test one
in `conftest`, which would delete the database under a server that is still answering.
"""

import shutil
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from huggingface_hub import snapshot_download
from openai import OpenAI

from mlx_omnia import GenerationOptions, ModelInput, paths
from mlx_omnia.server import Engine, create_app
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Job, Loader
from tests.server.openai_script import MODEL, loader


class Recording(Engine):
    """Keeps the jobs it hands out. What became of a request is the engine's record, and
    no dialect carries the notion — the `/admin` window on it is 33.2's."""

    def __init__(self, loader: Loader, daemon: Daemon) -> None:
        super().__init__(loader, daemon, Metrics())
        self.jobs: list[Job] = []

    async def submit(
        self,
        model_id: str,
        input: ModelInput,
        options: GenerationOptions,
        reservation: object | None = None,
        batch_limit: int | None = None,
    ) -> Job:
        job = await super().submit(model_id, input, options, reservation, batch_limit)
        self.jobs.append(job)
        return job


@pytest.fixture(scope="module", autouse=True)
def fresh_state() -> None:
    """One empty state directory per file, for the server the file starts."""
    root = paths.state_dir()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def daemon() -> Daemon:
    return Daemon()


@pytest.fixture(scope="module")
def engine(daemon: Daemon) -> Recording:
    return Recording(loader, daemon)


@pytest.fixture(scope="module")
def base_url(engine: Recording, daemon: Daemon, fresh_state: None) -> Iterator[str]:
    # On disk before the server binds: the dialects list the catalog, and the test that says
    # so must not be answering about what a previous run left in the cache.
    snapshot_download(MODEL)
    app = create_app(engine, host="127.0.0.1", daemon=daemon)

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
    """Temperature 0 explicitly: the dialect's default is 1.0, so an answer nobody pinned
    is drawn, and the tests below that compare two answers would be comparing two draws.

    What comes back is the two channels joined, because what these tests are about is the text
    the model wrote and not where the dialect files it. This checkpoint opens `<think>` on its
    first token, so a budget this small is spent inside the block and `content` is empty — a
    thinking model with no room to finish thinking answers nothing, which is the block meaning
    what it says.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    message = response.choices[0].message
    thought = (message.model_extra or {}).get("reasoning_content", "")
    assert isinstance(thought, str)
    written = thought + (message.content or "")
    assert written, "the model wrote on neither channel"
    return written


def post(base_url: str, **fields: object) -> httpx.Response:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
    return httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body | fields, timeout=60)


def answer(response: httpx.Response) -> str:
    """The two channels joined, for the same reason `ask` joins them: a budget spent inside
    the thinking block leaves `content` empty, and what these tests compare is the text the
    model drew."""
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    written = message.get("reasoning_content", "") + message["content"]
    assert isinstance(written, str) and written
    return written


def _health(base_url: str) -> dict[str, object]:
    payload = httpx.get(f"{base_url}/admin/health").json()
    assert isinstance(payload, dict)
    return payload


def resident(base_url: str) -> list[str]:
    """Residency through the catalog's own filter, which is the second `/admin` window on
    it besides `/admin/health`."""
    response = httpx.get(f"{base_url}/admin/models", params={"resident": True}, timeout=60)
    assert response.status_code == 200, response.text
    return [entry["id"] for entry in response.json()]


def asking(model: str, **fields: object) -> dict[str, object]:
    """The body of one request against a scripted model. `tools` is a field like any other
    here, so the test that asks what a request without them answers leaves it out."""
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "max_tokens": 64,
    }
    return body | fields


def offer(base_url: str, model: str, **fields: object) -> httpx.Response:
    url = f"{base_url}/api/openai/v1/chat/completions"
    return httpx.post(url, json=asking(model, **fields), timeout=60)


def sse(base_url: str, model: str, **fields: object) -> list[str]:
    """The `data:` payloads of one streamed request, `[DONE]` checked off and dropped."""
    url = f"{base_url}/api/openai/v1/chat/completions"
    body = asking(model, stream=True, **fields)
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        frames = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert frames[-1] == "[DONE]"
    return frames[:-1]
