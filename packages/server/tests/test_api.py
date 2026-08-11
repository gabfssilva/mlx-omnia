"""Gate P5: official OpenAI SDK against a real HTTP server process."""

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import TypeIs

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from huggingface_hub import snapshot_download
from openai import APIError, NotFoundError, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel

from sideros import (
    TEXT,
    ByteLevelBPE,
    Chat,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
    chat_template,
    load,
)
from sideros.parsers import FALLBACK, Segment, Segmenter
from sideros.schema import json_instruction
from sideros_server import Engine, create_app, prefixes
from sideros_server import engine as engine_module
from sideros_server.engine import Job, Loader
from sideros_server.store import Store

# A checkpoint with a chat template: /chat/completions takes a conversation, and what
# turns it into a prompt is the template. A base model has none and is refused (see
# `test_a_model_without_a_chat_template_is_refused`).
MODEL = "mlx-community/Qwen3-0.6B-4bit"
BASE_MODEL = "gpt2"

CALLER = "test/caller"
PAIR = "test/pair"
TRUNCATED = "test/truncated"
XML_CALLER = "test/xml-caller"
FLAKY = "test/flaky"
WRITER = "test/json-writer"
BREAKER = "test/schema-breaker"
MUTE = "test/no-json"
LEARNER = "test/schema-learner"
THINKER = "test/thinker"
"""Scripted models. What a checkpoint writes when it is offered a function is the
checkpoint's own decision, and a 0.6B asked for the weather is not a fixture: the tool
tests need the generated text pinned, and everything around it — the template, the
segmentation, the frames — stays real. A schema is the same: whether a 0.6B obeys one is
the checkpoint's business, and what these tests are about is what the server does with the
answer either way."""

SCRIPT = (
    "Let me check.",
    "<tool",
    '_call>\n{"name": "get_weather", "arguments": ',
    '{"city": "Paris"}}\n</tool',
    "_call>",
)
"""One generation, cut where a detokenizer would not: both markers straddle two pieces, so
a server that hands a piece out before it can tell hands out half a marker."""

PREAMBLE = SCRIPT[0]
CALL = "".join(SCRIPT[1:])
TIME_CALL = '<tool_call>\n{"name": "get_time", "arguments": {"zone": "Europe/Paris"}}\n</tool_call>'
RESULT = "22 C"
ANSWER = f"It is {RESULT}."

XML_SCRIPT = (
    "Let me check.\n",
    "<tool",
    "_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool",
    "_call>",
    "\nI will report back.",
)
"""What Qwen3.6 writes when it calls: Qwen's marker around `<function=...>` XML instead of
JSON, with both markers straddling two pieces like the one above. The sentence after the
call is what makes a server that held the envelope visible in the answer and not only in
the frames — what it hands back is the same characters with the call moved to the end."""

XML_ANSWER = "".join(XML_SCRIPT)

WEATHER: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
    "required": ["city", "celsius"],
    "additionalProperties": False,
}

JSON_SCHEMA: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {"name": "weather", "schema": WEATHER},
}
"""The dialect's non-strict form, typed as the SDK types it so the same value serves both
doors: the httpx body below and `create(response_format=...)`."""

WHOLE = '{"city": "Paris", "celsius": 22}'
FAULTY = '{"city": "Paris"}'

WRITTEN = ("Sure! ", '```json\n{"city": "Paris", ', '"celsius": 22}\n```')
"""What a model that was asked for JSON writes anyway: a line of prose, a code fence, and the
document cut across two pieces. Nothing here is malformed — what it is, is not `json.loads`-able
as it stands, which is what the client asked for."""


class Weather(BaseModel):
    """The pydantic model a client hands the SDK. `celsius` is a float and the document
    carries `22`: what is being read back is a document, not the characters it arrived in."""

    city: str
    celsius: float


SCRIPTS = {
    CALLER: SCRIPT,
    PAIR: (CALL, TIME_CALL),
    # An envelope the token budget cut in half: the call cannot be read, and the text the
    # model did write is the client's either way.
    TRUNCATED: ('<tool_call>\n{"name": "get_weat',),
    XML_CALLER: XML_SCRIPT,
    FLAKY: ("Half an ",),
    WRITER: WRITTEN,
    # A document that parses and breaks the schema, and one that is not a document at all:
    # the two failures level 1 can meet, and they are not the same error.
    BREAKER: (FAULTY,),
    MUTE: ("I would rather not.",),
    # A whole turn of a thinking model, with the closing marker straddling two pieces the way
    # a detokenizer hands them out: what the block is worth testing over is the seam.
    THINKER: ("<think>\nWeigh", "ing it.\n</think>", "\nParis."),
}

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time in a zone",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    },
]


@dataclass(frozen=True)
class Script:
    """A model whose generation is fixed text, handed out in the pieces it was given.

    Once the conversation carries a tool result it answers with it instead, which is what
    tells the second turn of a round trip from the first: the result only reaches here if
    the `tool` message became a turn the checkpoint's own template rendered.
    """

    pieces: tuple[str, ...]

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    fails: bool = False
    """Raises after handing out its pieces, which is the one way the worker meets an
    exception: an input the model refuses never becomes a job."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value))
        _, _, after = input.value.partition("<tool_response>")
        if result := after.partition("</tool_response>")[0].strip():
            meter.token()
            yield Segment("content", f"It is {result}.")
            return
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=input.value
        )
        for piece in self.pieces:
            # Counted like the loop counts: a double that hands out text without marking ids
            # leaves every number the dialect reports at zero, and a test reading one of them
            # would be measuring the double.
            meter.token()
            yield from segmenter.push(piece)
        yield from segmenter.flush()
        if self.fails:
            raise RuntimeError("the decode thread gave out")


@dataclass(frozen=True)
class Corrected:
    """A model that gets it right once it is told what was wrong.

    Which document it writes is read out of the rendered prompt rather than counted, and that
    is the point: a retry that resubmits the same conversation is the same generation — greedy
    lands on the same tokens — so a server that retried without feeding the failure back would
    get `FAULTY` again here, and the test would say so.
    """

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        meter.prefill(len(input.value))
        meter.token()
        yield Segment("content", WHOLE if "is required and missing" in input.value else FAULTY)


def template() -> ChatTemplate:
    """The checkpoint's own — what spells `<tools>`, `<tool_call>` and `<tool_response>`."""
    found = chat_template(Path(snapshot_download(MODEL)))
    assert found is not None
    return found


FIXTURE = Path(__file__).parents[3] / "packages" / "sideros" / "tests" / "fixtures"
QWEN36 = "mlx-community/Qwen3.6-35B-A3B-6bit"


def xml_template() -> ChatTemplate:
    """Qwen3.6's real template, out of the engine's own fixture: what the family is read off
    is the source, and a 35B is not a checkpoint this suite pulls to read one file of it."""
    golden = json.loads((FIXTURE / "chat_template.json").read_text(encoding="utf-8"))
    meta = golden["repos"][QWEN36]
    source = meta["template"]
    assert isinstance(source, str)
    return ChatTemplate.from_source(source, meta["special_tokens"])


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id == LEARNER:
        return CompositeModel(Corrected(), [ChatCapability(template())])
    script = SCRIPTS.get(model_id)
    if script is None:
        return load(model_id)
    spoken = xml_template() if model_id == XML_CALLER else template()
    return CompositeModel(Script(script, fails=model_id == FLAKY), [ChatCapability(spoken)])


class Recording(Engine):
    """Keeps the jobs it hands out. What became of a request is the engine's record, and
    no dialect carries the notion — the `/admin` window on it is 33.2's."""

    def __init__(self, loader: Loader) -> None:
        super().__init__(loader)
        self.jobs: list[Job] = []

    async def submit(self, model_id: str, input: ModelInput, options: GenerationOptions) -> Job:
        job = await super().submit(model_id, input, options)
        self.jobs.append(job)
        return job


@pytest.fixture(scope="module")
def engine() -> Recording:
    return Recording(loader)


@pytest.fixture(scope="module")
def base_url(engine: Recording, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    # On disk before the server binds: the dialects list the catalog, and the test that says
    # so must not be answering about what a previous run left in the cache.
    snapshot_download(MODEL)
    app = create_app(engine, Store(tmp_path_factory.mktemp("state") / "server.db"))

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


def test_health_says_who_is_answering_and_for_how_long(base_url: str) -> None:
    """The one route a key does not cover, and the only place a client that did not start the
    daemon can read the process it is talking to: the app draws it, and the CLI decides
    between talking to one and starting one."""
    payload = _health(base_url)
    assert payload["status"] == "ok"
    assert payload["pid"] == os.getpid(), "the stand runs the server in this process"
    uptime = payload["uptime"]
    assert isinstance(uptime, float) and uptime > 0
    assert payload["version"] == version("sideros")


def test_the_admin_routes_are_mounted_on_the_real_app(base_url: str) -> None:
    """Each `/admin` group lives in its own module and is wired in `create_app`. Their own
    suites mount the routers on a throwaway FastAPI, so nothing there would notice a group
    that never reaches the server the daemon actually serves."""
    for path in ("system", "models", "jobs", "state", "metrics", "config", "benches"):
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


def resident(base_url: str) -> list[str]:
    """Residency through the catalog's own filter, which is the second `/admin` window on
    it besides `/admin/health`."""
    response = httpx.get(f"{base_url}/admin/models", params={"resident": True}, timeout=60)
    assert response.status_code == 200, response.text
    return [entry["id"] for entry in response.json()]


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


def test_a_second_turn_reports_the_prefix_the_first_one_left(tmp_path: Path) -> None:
    """The trie seen from the dialect, on a real checkpoint: the second request repeats the
    first conversation and adds to it, so a stored cache covers its head. `cached_tokens` is
    the only thing that says so — a near miss and a hit produce the same answer at the same
    `prompt_tokens`, and differ only in what the prefill actually read.

    Its own stand, because the module's engine is built with no store and a store is what
    hands `prefix_budget` down (`engine.py:708`). An engine without one runs the cold path,
    which is what every test here that is not about reuse wants.

    A floor and not an equality: how much is covered depends on whether the template
    re-renders the assistant turn into the ids the model itself wrote, which is the
    checkpoint's business. What the daemon owes is the number, and that it is not zero.
    """
    store = Store(tmp_path / "server.db")
    opening = [{"role": "user", "content": "Name one river in Brazil."}]
    door = "/api/openai/v1/chat/completions"

    with TestClient(create_app(Engine(loader, store), store)) as running:
        first = running.post(
            door, json={"model": MODEL, "messages": opening, "max_tokens": 24, "temperature": 0}
        )
        assert first.status_code == 200, first.text
        continued = [
            *opening,
            {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]},
            {"role": "user", "content": "And one in Peru?"},
        ]
        second = running.post(
            door, json={"model": MODEL, "messages": continued, "max_tokens": 8, "temperature": 0}
        )

    assert second.status_code == 200, second.text
    usage = second.json()["usage"]
    assert usage["prompt_tokens_details"]["cached_tokens"] > 0, "the second turn prefilled cold"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_a_conversation_survives_the_daemon_it_was_warm_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the disk tier is for. The trie hangs on the model and dies with it, so without a
    file every conversation that was warm pays a whole prefill again after a restart.

    The restart is a second app over the same store and the same cache directory, which is
    what a restart is from the state's point of view — a new process finding what the last
    one left. The memory budget is one byte, so every entry is evicted the moment it is
    inserted and the second turn can only be answered off disk.
    """
    monkeypatch.setattr(prefixes, "CACHE", tmp_path / "prefixes")
    # A 0.6B's cache is ~115 KB a token, so the conversation below is a few megabytes and
    # sits under the floor a real one clears in its first thousand tokens. The floor is about
    # whether a file pays for itself, and that is not what this test is about.
    monkeypatch.setattr(engine_module, "_SPILL_FLOOR", 0)
    store = Store(tmp_path / "server.db")
    store.set_config({"prefix_cache_bytes": "1", "prefix_disk_bytes": str(4 * 1024**3)})
    opening = [{"role": "user", "content": "Name one river in Brazil."}]
    door = "/api/openai/v1/chat/completions"

    with TestClient(create_app(Engine(loader, store), store)) as first:
        answered = first.post(
            door, json={"model": MODEL, "messages": opening, "max_tokens": 24, "temperature": 0}
        )
        assert answered.status_code == 200, answered.text
    # Leaving the block is the shutdown, and a shutdown waits for the writes it started —
    # otherwise the daemon exits over a staging file and a row that never lands.
    assert store.prefix_files(MODEL), "nothing reached the disk to be found again"

    continued = [
        *opening,
        {"role": "assistant", "content": answered.json()["choices"][0]["message"]["content"]},
        {"role": "user", "content": "And one in Peru?"},
    ]
    with TestClient(create_app(Engine(loader, store), store)) as restarted:
        second = restarted.post(
            door, json={"model": MODEL, "messages": continued, "max_tokens": 8, "temperature": 0}
        )

    assert second.status_code == 200, second.text
    cached = second.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0, "the turn after the restart prefilled from cold"


def test_the_trie_turned_off_reports_zero_and_not_an_absent_field(tmp_path: Path) -> None:
    """`prefix_cache_bytes: 0` is the daemon with the trie off, and the same two turns then
    have nothing to reuse. The field is still written: absent reads as a server that does not
    carry it, and a client tuning a conversation to hit the cache needs to tell that from a
    miss."""
    store = Store(tmp_path / "server.db")
    store.set_config({"prefix_cache_bytes": "0"})
    opening = [{"role": "user", "content": "Name one river in Brazil."}]
    door = "/api/openai/v1/chat/completions"

    with TestClient(create_app(Engine(loader, store), store)) as running:
        first = running.post(
            door, json={"model": MODEL, "messages": opening, "max_tokens": 24, "temperature": 0}
        )
        assert first.status_code == 200, first.text
        continued = [
            *opening,
            {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]},
            {"role": "user", "content": "And one in Peru?"},
        ]
        second = running.post(
            door, json={"model": MODEL, "messages": continued, "max_tokens": 8, "temperature": 0}
        )

    assert second.status_code == 200, second.text
    assert second.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_include_usage_ends_the_stream_with_a_usage_frame(base_url: str) -> None:
    """The dialect's shape: one extra frame after the finish one and before `[DONE]`,
    carrying the whole request's usage and no choices. No frame before it carries usage —
    a client reading it off any chunk would be taking a partial count for the total."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        frames = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert frames[-1] == "[DONE]"
    last = json.loads(frames[-2])
    usage = last["usage"]
    assert last["choices"] == []
    assert usage["completion_tokens"] > 0
    # Not derivable from the other two: `total` is computed from them, so a streaming path
    # that lost the prompt count would satisfy the sum with a zero in it.
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert all("usage" not in json.loads(frame) for frame in frames[:-2])


def test_the_usage_frame_carries_what_the_turn_cost_in_seconds(base_url: str) -> None:
    """The dialect has no field for a rate, and a chat window that draws one would otherwise
    have to time the stream from outside — where the load, the queue and the prefill are one
    number. The extension rides on the frame that is already the request's total.

    `load_seconds` is not asserted: whether this request found the model resident depends on
    what the suite ran before it, and the key being present with either answer is the
    contract. What is asserted is that the key is there at all.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        frames = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    timings = json.loads(frames[-2])["x_sideros"]
    assert "load_seconds" in timings
    assert timings["ttft_seconds"] > 0
    assert timings["prefill_tokens_per_second"] > 0
    assert timings["tokens_per_second"] > 0
    assert timings["bytes_per_token"] > 0
    assert 0 < timings["ceiling_fraction"] < 1
    # Present and null, not absent: a turn that did not speculate says so, and a reader that
    # only ever sees this model would otherwise not know the key exists.
    assert timings["speculation"] is None


def test_the_official_sdk_accumulates_the_usage_frame(client: OpenAI) -> None:
    """The SDK's own accumulator over the stream: a frame it cannot fold in is a frame
    that reads as a malformed completion rather than as usage."""
    # A generation that ends on its own: the SDK's accumulator raises `LengthFinishReasonError`
    # on a truncated one, which is its own contract and not what this test is about.
    with client.chat.completions.stream(
        model=CALLER,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        temperature=0,
        stream_options={"include_usage": True},
    ) as stream:
        final = stream.get_final_completion()
    assert final.usage is not None
    assert final.usage.completion_tokens > 0
    assert final.choices[0].message.content


def test_a_client_that_closes_the_connection_leaves_a_cancelled_record(
    base_url: str, engine: Recording
) -> None:
    """Abandoning the stream used to be silent. The job that was cut mid-generation is
    recorded as cancelled, with the tokens it did emit."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a very long story."}],
        "max_tokens": 512,
        "stream": True,
    }
    url = f"{base_url}/api/openai/v1/chat/completions"
    with httpx.Client() as http, http.stream("POST", url, json=body, timeout=60) as response:
        for index, _line in enumerate(response.iter_lines()):
            if index >= 3:
                break

    job = engine.jobs[-1]
    deadline = time.time() + 10
    while job.state in ("queued", "running"):
        assert time.time() < deadline, "the worker never reached a terminal state"
        time.sleep(0.02)
    assert job.state == "cancelled"
    assert 0 < job.meter.completion_tokens < 512


def test_a_single_candidate_is_the_greedy_answer(base_url: str) -> None:
    """The two deterministic doors into the sampler — the zero of the dial and a cut that
    leaves one candidate — reach the same tokens the default greedy path reaches."""
    greedy_answer = answer(post(base_url, temperature=0))
    assert answer(post(base_url, temperature=1.4, top_k=1, seed=5)) == greedy_answer


def test_a_seed_replays_the_answer(base_url: str) -> None:
    seeded = answer(post(base_url, temperature=1.0, seed=1234, max_tokens=24))
    assert answer(post(base_url, temperature=1.0, seed=1234, max_tokens=24)) == seeded
    others = {answer(post(base_url, temperature=1.0, seed=s, max_tokens=24)) for s in (7, 99, 5150)}
    assert seeded not in others, "three other seeds all landed on the same answer"


def test_an_unseeded_request_is_drawn_not_argmaxed(base_url: str) -> None:
    """The dialect's default temperature is 1.0. If the server quietly used greedy instead,
    repeated unseeded requests would come back identical."""
    answers = {answer(post(base_url, max_tokens=24)) for _ in range(4)}
    assert len(answers) > 1


def test_sampling_reaches_the_engine(base_url: str) -> None:
    """A penalty strong enough to forbid every id already in the prompt or the answer:
    whatever comes back cannot repeat itself, which no default path would guarantee."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say the word banana ten times."}],
        "max_tokens": 24,
        "temperature": 0,
        "repetition_penalty": 1000.0,
    }
    response = httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body, timeout=60)
    penalized = answer(response)
    assert penalized != answer(post(base_url, temperature=0, max_tokens=24))


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"logit_bias": {"1": 2}}, "logit_bias"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"n": 2}, "n"),
        ({"temperature": -1}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"min_p": 1.0}, "min_p"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"stream_options": {"include_logprobs": True}}, "stream_options"),
        # Forcing a call is a constraint on decoding, and there is none here: answering
        # `auto` to a client that asked for `required` is the same lie as `logit_bias`.
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, "tool_choice"),
        # Retries of a check nobody asked for: the field would be read by nobody, which is
        # the one case a request-level schema cannot catch on its own.
        ({"max_schema_attempts": 2}, "max_schema_attempts"),
        (
            {"max_schema_attempts": 9, "response_format": {"type": "json_object"}},
            "max_schema_attempts",
        ),
        ({"response_format": {"type": "yaml"}}, "response_format"),
    ],
)
def test_a_field_we_do_not_honour_is_refused_by_name(
    base_url: str, client: OpenAI, fields: dict[str, object], named: str
) -> None:
    """Accepting `logit_bias` and ignoring it tells the client it was applied. The name has
    to be in the message, or the client is left guessing which of its fields we mean."""
    response = post(base_url, **fields)
    assert response.status_code == 400, response.text
    assert named in response.json()["error"]["message"]
    assert len(ask(client, "Hi", max_tokens=4)) > 0


def test_a_model_without_a_chat_template_is_refused(base_url: str) -> None:
    """A base model gets no guessed template, so the conversation has nowhere to go: a
    client error, not a 500 from the worker."""
    body = {"model": BASE_MODEL, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4}
    response = httpx.post(f"{base_url}/api/openai/v1/chat/completions", json=body, timeout=120)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_input"


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


def test_a_tool_call_round_trips_through_two_turns_of_the_official_sdk(client: OpenAI) -> None:
    """The whole path, judged by the SDK that will use it: the model is offered a function
    and answers with a call instead of text, the result of that call goes back in as a turn
    of its own, and the second answer is one only a model that was handed the result can
    give — `Script` reads it out of the `<tool_response>` the checkpoint's own template
    renders, so nothing but the conversion having worked puts it there."""
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Weather in Paris?"}]
    first = client.chat.completions.create(model=CALLER, messages=messages, tools=TOOLS)
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == PREAMBLE
    made = choice.message.tool_calls
    assert made is not None and len(made) == 1
    call = made[0]
    assert call.type == "function"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": call.id, "content": RESULT})
    second = client.chat.completions.create(model=CALLER, messages=messages, tools=TOOLS)
    assert second.choices[0].message.content == ANSWER
    assert second.choices[0].message.tool_calls is None
    assert second.choices[0].finish_reason == "stop"


def test_the_sdk_accumulator_builds_both_calls_out_of_the_stream(client: OpenAI) -> None:
    """What judges the frames is the SDK's own accumulator, not our reading of them. Two
    calls is what makes `index` load-bearing: the second frame's entry is merged into a list
    that already holds one, and an entry with no `index` raises there instead of folding
    in."""
    asked: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Weather and time in Paris?"}
    ]
    with client.chat.completions.stream(model=PAIR, messages=asked, tools=TOOLS) as stream:
        final = stream.get_final_completion()

    choice = final.choices[0]
    assert choice.finish_reason == "tool_calls"
    made = choice.message.tool_calls
    assert made is not None and len(made) == 2
    weather, clock = made
    assert weather.type == "function" and clock.type == "function"
    assert weather.function.name == "get_weather"
    assert json.loads(weather.function.arguments) == {"city": "Paris"}
    assert clock.function.name == "get_time"
    assert json.loads(clock.function.arguments) == {"zone": "Europe/Paris"}
    assert weather.id != clock.id


def test_tool_choice_none_neither_offers_the_tools_nor_reads_a_call_back(
    base_url: str, engine: Recording
) -> None:
    """`none` is honoured where it can be honoured: the tools never reach the prompt, so
    there is nothing to call rather than an instruction not to. And a turn nobody was
    offered a tool for is text — answering with a call would be answering with the one thing
    the client asked us not to do."""
    payload = offer(base_url, CALLER, tools=TOOLS, tool_choice="none").json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": PREAMBLE + CALL}

    job = engine.jobs[-1]
    assert isinstance(job.input, Chat)
    assert job.input.tools == ()


def test_a_request_without_tools_is_answered_with_the_envelope_it_wrote(base_url: str) -> None:
    """The suppression cannot charge a client that declared no tool: what the model wrote
    comes back whole, envelope and all, and no `tool_calls` are invented beside it.

    The frames are the model's segments and not the tokenizer's pieces, and that is the
    checkpoint's own doing rather than this route's: a model whose template declares a tool
    family segments on the way out, so an envelope leaves as one piece however many ids it
    took. What this route decides is only whether to *read* that channel — and with no tool
    offered it does not, so the text goes out as it came."""
    payload = offer(base_url, CALLER).json()
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": PREAMBLE + CALL}
    assert payload["choices"][0]["finish_reason"] == "stop"

    frames = [json.loads(frame) for frame in sse(base_url, CALLER)]
    deltas = [frame["choices"][0]["delta"] for frame in frames]
    assert deltas[0] == {"role": "assistant", "content": ""} and deltas[-1] == {}
    assert "".join(delta.get("content", "") for delta in deltas) == PREAMBLE + CALL
    assert not any("tool_calls" in delta for delta in deltas), "nobody offered a tool"
    finishes = [frame["choices"][0]["finish_reason"] for frame in frames]
    assert finishes[-1] == "stop" and set(finishes[:-1]) == {None}


def test_the_message_the_dialect_answers_with_is_a_message_it_accepts(base_url: str) -> None:
    """A client replays the answer it was given: the assistant turn goes back verbatim, the
    `index` inside the call and the `content: null` included. A field the request model
    refuses would 400 the second turn of every round trip that goes through an SDK."""
    answered = offer(base_url, CALLER, tools=TOOLS).json()["choices"][0]["message"]
    assert answered["content"] == PREAMBLE
    assert answered["tool_calls"][0]["index"] == 0

    replay = offer(
        base_url,
        CALLER,
        tools=TOOLS,
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            answered,
            {"role": "tool", "tool_call_id": answered["tool_calls"][0]["id"], "content": RESULT},
        ],
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["choices"][0]["message"]["content"] == ANSWER


def test_the_tools_and_the_call_a_result_answers_reach_the_conversation(
    base_url: str, engine: Recording
) -> None:
    """The frontier this stage owns: what the dialect's fields become in the `Chat` the
    engine is handed. The template renders from that and from nothing else, so a key the
    conversion drops is a key no checkpoint can put back — `tool_call_id` in particular,
    which the Qwen template never renders and only this can see."""
    history = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
    ]
    assert offer(base_url, CALLER, tools=TOOLS, messages=history).status_code == 200

    job = engine.jobs[-1]
    assert isinstance(job.input, Chat)
    assert job.input.tools == tuple(TOOLS)
    assert job.input.messages == (
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "content": RESULT, "tool_call_id": "call_1"},
    )


def test_an_envelope_the_budget_cut_in_half_comes_back_as_the_text_it_is(base_url: str) -> None:
    """Held as a possible call and it is not one: what the model wrote goes out as text. The
    silent failure this rules out is the opposite — an envelope suppressed and no call
    produced reaches the client as a model that chose to call nothing, which is exactly the
    shape of a correct refusal."""
    payload = offer(base_url, TRUNCATED, tools=TOOLS).json()
    choice = payload["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": SCRIPTS[TRUNCATED][0]}
    assert choice["finish_reason"] == "stop"


def test_a_checkpoint_whose_envelope_nothing_can_parse_answers_with_the_text_it_wrote(
    base_url: str,
) -> None:
    """Which family a checkpoint speaks is read off its chat template, and Qwen3.6's answer
    there is "none": it keeps Qwen's marker and fills it with XML. Sniffed from the generated
    text instead, that marker says Qwen — the envelope is held, no call comes out of it, and
    the client is handed the answer with the call moved to the end of it.

    Nothing is suppressed and nothing is an error: the turn is text, and text is what comes
    back, whole and in the pieces it was written in.

    On the reasoning channel, and that is the template's doing rather than this test's:
    Qwen3.6 writes `<think>` into the generation prompt, so the script starts inside the block
    and never closes it. Which channel the text came out on is not what is under test here —
    that it came out whole, in the pieces it was written in, is.
    """
    payload = offer(base_url, XML_CALLER, tools=TOOLS).json()
    choice = payload["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": XML_ANSWER,
    }
    assert choice["finish_reason"] == "stop"

    frames = [json.loads(frame) for frame in sse(base_url, XML_CALLER, tools=TOOLS)]
    assert [frame["choices"][0]["delta"] for frame in frames] == [
        {"role": "assistant", "content": ""},
        *({"reasoning_content": piece} for piece in XML_SCRIPT),
        {},
    ]


def test_a_generation_that_dies_mid_stream_is_an_error_and_not_a_finished_answer(
    client: OpenAI,
) -> None:
    """The failure a stream can hide: the response is already 200 and the frames already
    going out when the decode thread gives out. Closing with `finish_reason: "stop"` and
    `[DONE]` would be a completed answer, and the client would keep whatever text arrived as
    the whole of it — the SDK cannot tell that apart afterwards. A frame carrying `error` is
    what it raises on, which is why the failure travels as one."""
    stream = client.chat.completions.create(
        model=FLAKY,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=8,
        stream=True,
    )

    drawn: list[str] = []
    with pytest.raises(APIError) as failure:
        for chunk in stream:
            if piece := chunk.choices[0].delta.content:
                drawn.append(piece)

    assert "RuntimeError" in str(failure.value), "the reason is the daemon's own words"
    assert drawn == ["Half an "], "what did arrive is still the client's"


def test_a_generation_the_budget_cut_says_length_and_not_stop(client: OpenAI) -> None:
    """`length` is what an agent loop branches on to continue. Answering `stop` for a
    generation `max_tokens` cut hands it half a sentence as the final answer, and nothing else
    in the body says otherwise — the text is there either way.

    The real checkpoint and not a script: what decides is the count the loop wrote, and a
    double that yields text without counting ids has nothing to decide with. Both shapes,
    because the streaming one carries the reason in a frame of its own.
    """
    whole = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=1, temperature=0
    )
    assert whole.choices[0].finish_reason == "length"
    assert whole.usage is not None and whole.usage.completion_tokens == 1

    frames = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
        temperature=0,
        stream=True,
    )
    reasons = [chunk.choices[0].finish_reason for chunk in frames if chunk.choices]
    assert reasons[-1] == "length"


def test_a_generation_that_ended_on_its_own_still_says_stop(client: OpenAI) -> None:
    """The other half of the pair above, and what keeps it from being an assertion that
    every answer is truncated: the same checkpoint with a budget it does not reach."""
    answered = client.chat.completions.create(
        model=CALLER, messages=[{"role": "user", "content": "Hi"}], max_tokens=64
    )
    assert answered.choices[0].finish_reason == "stop"


def test_json_object_answers_with_the_document_and_not_with_the_text_around_it(
    base_url: str, engine: Recording
) -> None:
    """Level 1 end to end: the schema — here only the demand for JSON — enters the prompt as a
    turn of its own, and what comes back is the value that was checked.

    The prose and the fence stay out of the answer. A client that asked for `json_object` and
    got `Sure! ```json…``` ` back was handed something `json.loads` refuses, and the document
    is the one thing this server can promise: it is what it validated.

    The instruction is the engine's own `json_instruction` and not a second copy: what is asked
    for and what `validate` enforces cannot be two readings of the same vocabulary.
    """
    response = offer(base_url, WRITER, response_format={"type": "json_object"})
    payload = response.json()
    assert response.status_code == 200, response.text
    content = payload["choices"][0]["message"]["content"]
    assert json.loads(content) == {"city": "Paris", "celsius": 22}
    assert "Sure!" not in content
    assert payload["schema_attempts"] == 1, "the default is one generation, and it is visible"

    conversation = engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert conversation.messages[-1] == {"role": "system", "content": json_instruction(None)}


def test_a_json_schema_answer_validates_into_the_model_that_asked_for_it(client: OpenAI) -> None:
    """The typed round trip through the official SDK at level 1: the schema goes into the
    prompt and the answer is measured against it afterwards, which is a check that can fail —
    what the client gets when it does not is a document its own model accepts. The guarantee
    is `strict`, and it is the test below."""
    completion = client.chat.completions.create(
        model=WRITER,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=64,
        response_format=JSON_SCHEMA,
    )
    content = completion.choices[0].message.content
    assert content is not None
    assert Weather.model_validate_json(content) == Weather(city="Paris", celsius=22)


def test_the_sdk_s_parse_helper_comes_back_typed_off_a_decode_that_could_not_violate(
    client: OpenAI,
) -> None:
    """Level 3 through the door the SDK actually uses, against a real checkpoint.

    `parse()` puts `strict: true` on every param it builds out of a pydantic model, so this is
    the helper's own path and not a hand-written body. What makes it typed is the mask: the
    schema is compiled against this checkpoint's own token table and the ids that would break
    it are at -inf before the draw. The same request without `strict` was measured on this
    checkpoint and comes back 422 `malformed_json` — a 0.6B asked nicely writes prose — so what
    is being read here is the guarantee and not the model's good manners.

    `finish_reason` is checked because the SDK raises on `length` instead of parsing: a run the
    budget cut is not an answer this level promised anything about. And the content is checked
    against the schema's own keys, because the grammar is what keeps a reasoning block, a fence
    and a field nobody declared out of a document that would otherwise carry all three.
    """
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=64,
        temperature=0,
        response_format=Weather,
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "stop"
    assert isinstance(choice.message.parsed, Weather)
    assert choice.message.content is not None
    assert sorted(json.loads(choice.message.content)) == ["celsius", "city"]


def test_a_schema_the_grammar_will_not_take_is_a_400_in_the_compiler_s_own_words(
    base_url: str,
) -> None:
    """A guarantee this server cannot give is refused and never quietly demoted: dropping to
    level 1 here would answer a client that asked for a decode that cannot violate the schema
    with a check against a schema keyword nobody enforces.

    The message is the compiler's — `Unimplemented keys: ["uniqueItems"]` is a reason where
    "grammar error" is not, and it is what tells the client which keyword to drop. Sending the
    same schema without `strict` is the way through, and it answers.

    Against the real checkpoint and not a scripted model: a refusal is only the compiler's if
    there was a token table to refuse against, and a double has none (the test below).
    """
    unique = {
        "type": "json_schema",
        "json_schema": {
            "name": "cities",
            "schema": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "strict": True,
        },
    }
    refused = offer(base_url, MODEL, response_format=unique)
    assert refused.status_code == 400, refused.text
    error = refused.json()["error"]
    assert error["code"] == "grammar_refused"
    assert "uniqueItems" in error["message"]


def test_strict_without_a_schema_is_refused_rather_than_read_as_json_object(
    base_url: str,
) -> None:
    """The one shape of `strict` that means nothing: there is no schema for decoding to be
    constrained by, and honouring it as "answer in JSON" would call a demand a guarantee."""
    empty = {"type": "json_schema", "json_schema": {"name": "weather", "strict": True}}
    refused = offer(base_url, WRITER, response_format=empty)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "strict_without_schema"


def test_strict_and_tools_are_refused_together_because_a_grammar_leaves_no_room_for_a_call(
    base_url: str,
) -> None:
    """The one combination `strict` makes impossible rather than expensive: the mask allows
    the schema's ids from the first token, so an offered function can never be called. A 200
    with the tools silently uncallable is the answer this refusal exists instead of.

    `tool_choice: "none"` is not this case — the tools never enter the prompt, so there is
    nothing being dropped — and the test above it proves the non-strict pair still works.
    """
    strict = {
        "type": "json_schema",
        "json_schema": {"name": "weather", "schema": WEATHER, "strict": True},
    }
    refused = offer(base_url, MODEL, tools=TOOLS, response_format=strict)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "strict_with_tools"

    ignored = offer(
        base_url, MODEL, tools=TOOLS, tool_choice="none", response_format=strict, temperature=0
    )
    assert ignored.status_code == 200, ignored.text


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    base_url: str,
) -> None:
    """A scripted model is not in any catalog and holds no tokenizer, so there is no token
    table to compile against. What the client must not get is the answer anyway: `strict` asks
    for a guarantee, and a 200 carrying a free decode is that guarantee broken silently."""
    strict = {
        "type": "json_schema",
        "json_schema": {"name": "weather", "schema": WEATHER, "strict": True},
    }
    refused = offer(base_url, WRITER, response_format=strict)
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "not_constrainable"


def test_a_document_that_breaks_the_schema_is_a_named_error_and_never_a_success(
    base_url: str,
) -> None:
    """The failure this level exists to make impossible: JSON that parses, a document that
    does not conform, and a 200 with it inside. The path is in the message because a client
    that is told "invalid" and not *where* is left diffing document against schema by hand.

    422 and not a 5xx: the SDKs retry a 5xx by themselves, and a whole generation bought behind
    the client's back is exactly the cost this level is supposed to show.
    """
    response = offer(base_url, BREAKER, response_format=JSON_SCHEMA)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "schema_violation"
    assert "$.celsius is required and missing" in error["message"]
    assert "after 1 generation" in error["message"], "what it cost is part of the answer"


def test_an_answer_with_no_json_in_it_is_the_other_failure_and_says_which(base_url: str) -> None:
    """A model that answered in prose is not a model that broke the schema, and the two are
    not one error: what a client does about them differs, and only one of them has a path."""
    response = offer(base_url, MUTE, response_format={"type": "json_object"})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "malformed_json"


def test_the_second_attempt_is_the_client_s_to_buy_carries_the_failure_and_is_counted(
    base_url: str, engine: Recording
) -> None:
    """The retry policy, whole: nothing is retried by default, the client raises the ceiling,
    and every generation it cost is on the answer — in `schema_attempts` and in `usage`.

    What goes into the second prompt is what the first one wrote and what was wrong with it.
    A retry over the same conversation would be the same generation, and `Corrected` answers
    off the prompt precisely so that a server that resubmitted it blind would fail here.
    """
    alone = offer(base_url, LEARNER, response_format=JSON_SCHEMA)
    assert alone.status_code == 422, "one generation is the default, and it does not retry"

    response = offer(base_url, LEARNER, response_format=JSON_SCHEMA, max_schema_attempts=2)
    payload = response.json()
    assert response.status_code == 200, response.text
    assert json.loads(payload["choices"][0]["message"]["content"]) == json.loads(WHOLE)
    assert payload["schema_attempts"] == 2
    assert payload["usage"]["completion_tokens"] == 2, "both generations are on the bill"

    conversation = engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert conversation.messages[-2] == {"role": "assistant", "content": FAULTY}
    told = conversation.messages[-1]["content"]
    assert isinstance(told, str) and "$.celsius is required and missing" in told


def test_a_stream_checks_what_it_already_sent_and_fails_with_an_error_frame(
    client: OpenAI,
) -> None:
    """A stream has one pass: the frames are gone by the time the document can be checked, so
    the violation travels the way a generation that died travels — a frame carrying `error`,
    which is the only way to fail a request that already answered 200. Closing with
    `finish_reason: "stop"` instead would hand the client a document it believes was checked."""
    stream = client.chat.completions.create(
        model=BREAKER,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        max_tokens=64,
        response_format=JSON_SCHEMA,
        stream=True,
    )
    drawn: list[str] = []
    with pytest.raises(APIError) as failure:
        for chunk in stream:
            if piece := chunk.choices[0].delta.content:
                drawn.append(piece)

    assert "$.celsius" in str(failure.value)
    assert drawn == [FAULTY], "what did arrive is still the client's"


def test_a_stream_cannot_buy_a_second_attempt_and_says_so(base_url: str) -> None:
    """The one combination that cannot be honoured at all: a second attempt is a second
    generation, and the first one's frames have already left."""
    refused = offer(
        base_url, WRITER, response_format=JSON_SCHEMA, max_schema_attempts=2, stream=True
    )
    assert refused.status_code == 400, refused.text
    message = refused.json()["error"]["message"]
    assert "max_schema_attempts" in message and "stream" in message


def test_a_turn_that_called_something_is_not_an_answer_to_check(base_url: str) -> None:
    """`response_format` is about the answer, and a turn that called a function has not
    answered yet: checking the preamble against the schema would turn every tool call under a
    `response_format` into a 422, which is refusing the one thing the tools were offered for."""
    payload = offer(base_url, CALLER, tools=TOOLS, response_format=JSON_SCHEMA).json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == PREAMBLE
    assert payload["schema_attempts"] == 1


def test_the_thinking_block_is_a_field_of_its_own_and_never_the_answer(base_url: str) -> None:
    """A11's channel, in the field this dialect has for it — the one DeepSeek, vLLM and Ollama
    answer with over this same route.

    Without it the block goes out as the answer: a client draws `<think>` and everything under
    it as what the model said, and the answer itself starts arriving hundreds of tokens into
    the turn. The markers stay out of both fields, and out of the field that names the channel
    they would be the model quoting itself.
    """
    message = offer(base_url, THINKER).json()["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": "\nParis.",
        "reasoning_content": "\nWeighing it.\n",
    }

    frames = [json.loads(frame) for frame in sse(base_url, THINKER)]
    # The two reasoning frames are the two pieces the block was written in — the seam of
    # `</think>` is what the second one closes, and it leaves as text with no marker in it.
    assert [frame["choices"][0]["delta"] for frame in frames] == [
        {"role": "assistant", "content": ""},
        {"reasoning_content": "\nWeigh"},
        {"reasoning_content": "ing it.\n"},
        {"content": "\nParis."},
        {},
    ]


def test_a_turn_that_thinks_nothing_carries_no_reasoning_at_all(base_url: str) -> None:
    """The field is the block's, not every turn's: a checkpoint that writes no block answers
    exactly what it answered before there was a channel to name."""
    message = offer(base_url, MUTE).json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "I would rather not."}

    frames = [json.loads(frame) for frame in sse(base_url, MUTE)]
    assert [frame["choices"][0]["delta"] for frame in frames] == [
        {"role": "assistant", "content": ""},
        {"content": "I would rather not."},
        {},
    ]
