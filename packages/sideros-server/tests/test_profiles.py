"""Profiles: `model:profile` as a served id, and what naming one changes in a request.

No checkpoint is loaded here. What a profile has to reach is the sampler and the chat
template, and both run for real — `sideros.sampler` over a model whose logits are a fixed
table, and the checkpoint-side `ChatCapability` over a ChatML template — which is what
makes these assertions statements about the wiring rather than about a small model's
willingness to obey an instruction.

The rendered prompt is otherwise unobservable from outside: it is built inside the model,
behind `stream`. `Echo` hands it back, so the system prompt a profile carries can be read
where it actually lands.
"""

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi.testclient import TestClient

from sideros import (
    TEXT,
    Chat,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    KVCache,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
)
from sideros.suppress import Segment
from sideros_server import Engine, catalog, create_app
from sideros_server.store import Store

MODEL = "mlx-community/Qwen3-0.6B-4bit"
DENSE = "Qwen/Qwen3-0.6B"
"""Ids of the catalog fixture: what they name is never downloaded, and what the tests read
off them is that an id carrying a `/` survives the route's path converter."""

TINY = "test/tiny"
ECHO = "test/echo"

SYSTEM = "Answer only with the word BANANA."

VOCAB = 8

TEMPLATE = (
    r"{% for message in messages %}"
    r"{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    r"{% endfor %}"
    r"{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
"""ChatML, which is the shape of the template every checkpoint in the suite ships. Raw:
the `\\n` is Jinja's escape inside a Jinja string, the way a checkpoint writes it."""

CONFIG: dict[str, object] = {"model_type": "qwen3", "max_position_embeddings": 4096}


class TinyLM(nn.Module):
    """Logits from a fixed table, one row per token. Greedy over it is a fixed cycle and a
    draw at temperature 2 is not, which is the whole difference the sampling test reads —
    the spread is deliberately small so no token dominates the flattened distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.table = mx.array(
            [
                [((row * 3 + column * 5) % 7) * 0.25 for column in range(VOCAB)]
                for row in range(VOCAB)
            ]
        )

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.table[ids]


class TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [sum(text.encode()) % VOCAB]

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ord("a") + token for token in ids)


class Echo:
    """A model whose answer is the prompt it was handed."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


def template() -> ChatTemplate:
    return ChatTemplate.from_source(TEMPLATE)


def tiny() -> CompositeModel[Text, Segment, GenerationOptions]:
    """The production chain: a conversation enters through `ChatCapability`, which renders
    it, and the text goes to the model the way a checkpoint's would."""
    return CompositeModel(
        TextLanguageModel(TinyLM(), TinyTokenizer()), [ChatCapability(template())]
    )


def echo() -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(Echo(), [ChatCapability(template())])


@dataclass
class Loader:
    """Records every id it is asked for: which name reached the loader is the whole claim
    of the resolution rule, and a 200 by itself does not say which one did."""

    models: Mapping[str, LanguageModel[ModelInput]]
    asked: list[str] = field(default_factory=list)

    def __call__(self, model_id: str) -> LanguageModel[ModelInput]:
        self.asked.append(model_id)
        model = self.models.get(model_id)
        if model is None:
            raise FileNotFoundError(model_id)
        return model


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "server.db")


@pytest.fixture
def local(tmp_path: Path) -> str:
    """A local checkpoint whose path carries a `:` — the case that makes the split rule
    conditional, since a Hub id never has one."""
    directory = tmp_path / "q:2"
    directory.mkdir()
    return str(directory)


@pytest.fixture
def loader(local: str) -> Loader:
    return Loader({TINY: tiny(), ECHO: echo(), local: tiny()})


@pytest.fixture
def client(store: Store, loader: Loader) -> Iterator[TestClient]:
    """The app the daemon serves, not a router mounted on a throwaway `FastAPI`: what the
    profile routes have to survive is sharing `/admin/models` with the catalog's."""
    with TestClient(create_app(Engine(loader), store)) as running:
        yield running


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The scan reads two module constants; the user's own caches are never touched."""
    root = tmp_path / "hub"
    monkeypatch.setattr(catalog, "HUB_CACHE", root)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    # Both caches are keyed by id and the ids here repeat across tests, each over its own
    # `tmp_path`: without this, the second test to ask reads the first one's disk.
    catalog.context_of.cache_clear()
    catalog.defaults_of.cache_clear()
    return root


def installed(hub: Path, model_id: str, generation: Mapping[str, object] | None = None) -> None:
    repository = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(CONFIG))
    if generation is not None:
        (snapshot / "generation_config.json").write_text(json.dumps(generation))
    (snapshot / "model.safetensors").write_bytes(b"\0" * 64)
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("head")


def put(client: TestClient, model: str, name: str, **body: object) -> None:
    response = client.put(f"/admin/models/{model}/profiles/{name}", json=body)
    assert response.status_code == 200, response.text


_DEADLINE = 30.0
"""A deadline that holds, unlike the `timeout=` the `TestClient` accepts and ignores. Only the
chat request needs one: it waits on the engine's queue from inside the handler, so a worker
that never answers hangs the session instead of failing this test."""


def ask(client: TestClient, model: str, **fields: object) -> str:
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 32,
    }
    answered: list[httpx.Response] = []

    def post() -> None:
        answered.append(client.post("/api/openai/v1/chat/completions", json=body | fields))

    thread = threading.Thread(target=post, daemon=True)
    thread.start()
    thread.join(_DEADLINE)
    assert answered, f"the request did not answer within {_DEADLINE}s"
    response = answered[0]
    assert response.status_code == 200, response.text
    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    return content


def test_a_profile_is_written_read_replaced_and_deleted_under_its_model(
    client: TestClient, store: Store
) -> None:
    """The sub-resource of a model, on the path it shares with the catalog's
    `{model_id:path}` — that one matches slashes, so mounted in the wrong order it answers
    here first and reads `.../profiles/code` as part of the model's name. A `GET` that comes
    back with the profile is what says the order is right.

    A `PUT` replaces the profile rather than patching it: what the second one omits is gone,
    which is the only reading under which the body a client sends is the profile.
    """
    url = f"/admin/models/{MODEL}/profiles/code"
    created = client.put(
        url,
        json={"sampling": {"temperature": 0.2, "top_k": 5}, "system_prompt": "be terse"},
    )
    assert created.status_code == 200, created.text
    assert created.json() == {
        "model": MODEL,
        "name": "code",
        "sampling": {
            "temperature": 0.2,
            "top_p": None,
            "top_k": 5,
            "min_p": None,
            "repetition_penalty": None,
            "seed": None,
            "reasoning_effort": None,
            "reasoning_budget": None,
        },
        "system_prompt": "be terse",
    }
    assert client.get(url).json() == created.json()

    row = store.profile(MODEL, "code")
    assert row is not None
    assert json.loads(row.sampling) == {"temperature": 0.2, "top_k": 5}

    assert client.put(url, json={"sampling": {"seed": 3}}).status_code == 200
    replaced = client.get(url).json()
    assert replaced["sampling"]["seed"] == 3
    assert replaced["sampling"]["temperature"] is None
    assert replaced["system_prompt"] is None

    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404
    assert client.delete(url).status_code == 404


def test_a_body_naming_the_template_is_refused_by_name(client: TestClient) -> None:
    """A profile that accepted `template` and never rendered it would be telling the client,
    wrongly, that its template is in use — the rule the dialect already applies to
    `logit_bias`. The name has to be in the message or the client is left guessing."""
    response = client.put(
        f"/admin/models/{MODEL}/profiles/code",
        json={"sampling": {}, "template": "{{ messages }}"},
    )
    assert response.status_code == 400, response.text
    assert "template" in response.json()["error"]["message"]


def test_a_profile_name_carrying_a_colon_is_refused(client: TestClient) -> None:
    """Resolution splits at the last `:`, so `a:b` is a profile no request could name:
    `model:a:b` goes looking for the profile `b` of a model called `model:a`. Storing one
    is storing something unreachable."""
    response = client.put(f"/admin/models/{MODEL}/profiles/a:b", json={})
    assert response.status_code == 400, response.text
    assert client.get(f"/admin/models/{MODEL}/profiles/a:b").status_code == 404


def test_two_profiles_are_listed_beside_the_bare_id_and_one_of_an_unknown_model_is_not(
    client: TestClient, hub: Path
) -> None:
    """The dialect has no field for a profile, so an id a client cannot see is a profile it
    cannot select: a model with two of them answers to three names. A profile whose model is
    not in the catalog is not one of them — there is nothing to serve it with."""
    installed(hub, MODEL)
    installed(hub, DENSE)
    put(client, MODEL, "strict", sampling={"temperature": 0})
    put(client, MODEL, "code", sampling={"top_k": 5})
    put(client, "ghost/model", "x", sampling={})

    listed = client.get("/api/openai/v1/models").json()["data"]
    assert [entry["id"] for entry in listed] == [
        DENSE,
        MODEL,
        f"{MODEL}:code",
        f"{MODEL}:strict",
    ]


def test_a_request_naming_a_profile_is_sampled_by_it(client: TestClient) -> None:
    """Temperature 0 against temperature 2 with a seed, over a request that carries neither:
    the knobs have to come from the profile, since there is nowhere else for them to come
    from. Were the profile dropped, both would fall to the dialect's default — 1.0 and
    unseeded — and neither answer would repeat."""
    put(client, TINY, "cold", sampling={"temperature": 0})
    put(client, TINY, "hot", sampling={"temperature": 2, "seed": 7})

    cold = ask(client, f"{TINY}:cold")
    hot = ask(client, f"{TINY}:hot")
    assert len(cold) == 32, "one letter per token, so a short answer is a stop nobody asked for"
    assert cold == ask(client, f"{TINY}:cold")
    assert hot == ask(client, f"{TINY}:hot")
    assert hot != cold


def test_a_knob_the_request_sends_wins_over_the_profiles(client: TestClient) -> None:
    """A request that names a temperature means it, whatever the profile it also named says.
    The dialect's defaults are values like any other, so what tells an unset field from an
    explicit one is the request's own field set and never the value."""
    put(client, TINY, "cold", sampling={"temperature": 0})
    put(client, TINY, "hot", sampling={"temperature": 2, "seed": 7})

    greedy = ask(client, f"{TINY}:cold")
    assert ask(client, f"{TINY}:hot", temperature=0) == greedy
    assert ask(client, f"{TINY}:hot") != greedy


def test_a_request_naming_no_knob_is_sampled_by_what_the_checkpoint_declares(
    client: TestClient, hub: Path
) -> None:
    """`generation_config.json` is how the people who trained a checkpoint say it wants to
    be sampled, and a request that names nothing gets it. `do_sample: false` is the argmax,
    so the answer repeats — and the same request against the same model with the file
    removed does not, which is what says the repetition is the file's doing and not the
    dialect's."""
    installed(hub, TINY, generation={"do_sample": False, "temperature": 0.6})

    greedy = ask(client, TINY)
    assert len(greedy) == 32, "one letter per token, so a short answer is a stop nobody asked for"
    assert greedy == ask(client, TINY)


def test_a_checkpoint_that_declares_nothing_leaves_the_dialects_defaults(
    client: TestClient, hub: Path
) -> None:
    """The mutation of the test above: same model, same request, no `generation_config.json`
    — and the dialect's own 1.0 unseeded is what samples it, which does not repeat."""
    installed(hub, TINY)

    assert ask(client, TINY) != ask(client, TINY)


def test_a_profile_wins_over_what_the_checkpoint_declares(client: TestClient, hub: Path) -> None:
    """The checkpoint says how it was meant to be sampled in general; a profile is written
    for a job, and the more specific of the two is the one that stands."""
    installed(hub, TINY, generation={"do_sample": False, "temperature": 0.6})
    put(client, TINY, "hot", sampling={"temperature": 2, "seed": 7})

    greedy = ask(client, TINY)
    hot = ask(client, f"{TINY}:hot")
    assert hot != greedy
    assert hot == ask(client, f"{TINY}:hot")


def test_a_knob_the_request_sends_wins_over_what_the_checkpoint_declares(
    client: TestClient, hub: Path
) -> None:
    """The lowest of the three levels is the checkpoint's, so a client that names a
    temperature means it there too."""
    installed(hub, TINY, generation={"do_sample": False, "temperature": 0.6})

    greedy = ask(client, TINY)
    assert ask(client, TINY, temperature=2) != greedy


def test_the_profiles_system_prompt_reaches_the_template(client: TestClient) -> None:
    """The render happens inside the model, behind `ChatCapability`. What comes back here is
    the prompt the model was handed, and it has to be the same text the template produces
    for the conversation the profile should have built: the system turn first, the client's
    turn after it."""
    put(client, ECHO, "banana", system_prompt=SYSTEM)

    prompted = ask(client, f"{ECHO}:banana")
    # Spelled out once, so that a template rendering nothing at all cannot satisfy the two
    # comparisons below by rendering nothing on both sides of them.
    assert prompted.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
    assert prompted == template().render(
        Chat(({"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hi"}))
    )
    assert ask(client, ECHO) == template().render(Chat(({"role": "user", "content": "Hi"},)))


def test_a_system_message_from_the_client_replaces_the_profiles(client: TestClient) -> None:
    """Two system turns is a conversation the template has to pick between, and which one it
    picks is the template's business. The request is what says what the model is."""
    put(client, ECHO, "banana", system_prompt=SYSTEM)

    rendered = ask(
        client,
        f"{ECHO}:banana",
        messages=[
            {"role": "system", "content": "You are a cat."},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert rendered.startswith("<|im_start|>system\nYou are a cat.<|im_end|>\n")
    assert rendered == template().render(
        Chat(
            (
                {"role": "system", "content": "You are a cat."},
                {"role": "user", "content": "Hi"},
            )
        )
    )
    assert SYSTEM not in rendered


def test_a_local_path_carrying_a_colon_resolves_whole(
    client: TestClient, loader: Loader, local: str
) -> None:
    """Why the split is conditional: a Hub id has no `:` and a path may. The suffix only
    wins when it names a profile *of the head*, so a profile called `2` on another model
    must not turn `~/q:2` into the profile `2` of `~/q`."""
    put(client, TINY, "2", sampling={"temperature": 0})

    assert len(ask(client, local)) == 32
    assert loader.asked == [local]


def test_a_profile_that_does_not_exist_is_refused_instead_of_the_bare_model(
    client: TestClient, loader: Loader
) -> None:
    """A `model:typo` that generates with the defaults is the failure nobody notices. No
    profile matched, so the name was never split: what was asked for is a checkpoint by that
    whole name, and there is none."""
    put(client, TINY, "cold", sampling={"temperature": 0})

    response = client.post(
        "/api/openai/v1/chat/completions",
        json={
            "model": f"{TINY}:typo",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "model_not_found"
    assert loader.asked == [f"{TINY}:typo"]
    # And the model under the name is available, so the refusal is about the suffix.
    assert len(ask(client, TINY, max_tokens=4)) == 4
