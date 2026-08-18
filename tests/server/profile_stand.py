"""What the profile suites generate with, and what they read the stored rows through.

No checkpoint is loaded: what a profile has to reach is the sampler and the chat template,
and both run for real — `mlx_omnia.sampler` over a model whose logits are a fixed table, and
the checkpoint-side `ChatCapability` over a ChatML template — which is what makes those
assertions statements about the wiring rather than about a small model's willingness to obey
an instruction.

The rendered prompt is otherwise unobservable from outside: it is built inside the model,
behind `stream`. `Echo` hands it back, so the system prompt a profile carries can be read
where it actually lands.
"""

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
from fastapi.testclient import TestClient

from mlx_omnia import (
    TEXT,
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
    paths,
)
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.parsers import Segment

SCANNER = import_module("mlx_omnia.server.services.catalog.scan")
"""The module holding the two cache constants the scan reads. Reached by name because the
package re-exports a `scan` *function* under that same attribute, so `catalog.scan` is not
the module and rebinding a constant on the package would leave the scan on the real cache."""

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

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.table[ids]


class TinyTokenizer:
    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter([sum(whole.encode()) % VOCAB])

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


def installed(hub: Path, model_id: str, generation: Mapping[str, object] | None = None) -> None:
    repository = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(CONFIG))
    if generation is not None:
        (snapshot / "generation_config.json").write_text(json.dumps(generation))
    # A real one-tensor shard: admission weighs the checkpoint off the header before it lets
    # the load through, so arbitrary bytes here are a request refused rather than a model.
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [16], "data_offsets": [0, 64]}}
    ).encode()
    (snapshot / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0" * 64
    )
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("head")


def put(client: TestClient, model: str, name: str, **body: object) -> None:
    response = client.put(f"/admin/models/{model}/profiles/{name}", json=body)
    assert response.status_code == 200, response.text


def sample(client: TestClient, model: str, **knobs: object) -> None:
    response = client.put(f"/admin/models/{model}/sampling", json=knobs)
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


def profile_row(model: str, name: str) -> tuple[str, str | None] | None:
    """`(sampling, system_prompt)` straight out of the file, for a claim about what was
    stored rather than about what the route answered with."""
    with closing(sqlite3.connect(paths.server_db())) as connection:
        row = connection.execute(
            "SELECT sampling, system_prompt FROM profiles WHERE model = ? AND name = ?",
            (model, name),
        ).fetchone()
    if row is None:
        return None
    sampling, system_prompt = row
    return str(sampling), None if system_prompt is None else str(system_prompt)


def settings_row(model: str) -> tuple[str, int | None]:
    """`(sampling, max_concurrent_requests)`, with the empty defaults an absent row means."""
    with closing(sqlite3.connect(paths.server_db())) as connection:
        row = connection.execute(
            "SELECT sampling, max_concurrent_requests FROM model_settings WHERE model = ?",
            (model,),
        ).fetchone()
    if row is None:
        return "{}", None
    sampling, limit = row
    return str(sampling), None if limit is None else int(limit)
