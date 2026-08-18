"""The fakes the engine suite generates over, and the two seeds its config needs.

Shared rather than duplicated because the engine tests are split by what they are about —
scheduling, batching, grammars — and every one of them needs a model that streams.
"""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from importlib import import_module
from pathlib import Path
from typing import TypeIs

import pytest

from mlx_omnia import (
    TEXT,
    GenerationOptions,
    ModelInput,
    ModelSignature,
    Text,
    paths,
)
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.main import migrate
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.services import catalog

_SCAN = import_module("mlx_omnia.server.services.catalog.scan")
"""The submodule and not `catalog.scan`, which is the re-exported function of that name."""


def caches_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The two caches the scan reads, pointed away from the machine's own, and returns the
    hub. Both names are patched: `catalog` re-exports the constants by value, so the package
    attribute the app watches and the module attribute the scan reads are two objects."""
    hub, quantized = tmp_path / "hub", tmp_path / "quantized"
    for module in (catalog, _SCAN):
        monkeypatch.setattr(module, "HUB_CACHE", hub)
        monkeypatch.setattr(module, "QUANTIZED_CACHE", quantized)
    return hub


class FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[tuple[Text, GenerationOptions]] = []

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        self.calls.append((input, options))
        yield Segment("content", input.value[: options.max_tokens])


class AsciiTokenizer:
    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter(whole.encode())

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)


_STOP = 256
_PIECES = {value: bytes([value]) for value in range(256)} | {_STOP: b"<|end|>"}


class ByteTokenizer:
    """One id per byte plus one that ends a turn. Small enough to build a table over in a
    test and real enough for a grammar: JSON is bytes, and llguidance walks these."""

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        whole = text if isinstance(text, str) else "".join(text)
        return iter(list(whole.encode("utf-8")))

    def decode_bytes(self, ids: list[int]) -> bytes:
        # `KeyError` and not an empty piece: it is what an id past the tokenizer's own count
        # raises, and a padded head is made of those.
        return b"".join(_PIECES[identifier] for identifier in ids)


class ConstrainedLanguageModel(FakeLanguageModel):
    """A model a grammar can be built over. The two things `Vocabulary` needs that no model
    protocol declares live where the real facades put them: the checkpoint's own tokenizer,
    and the stop set the load resolved."""

    tokenizer = ByteTokenizer()
    stop = (_STOP,)


CITY: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
UNIQUE: dict[str, object] = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
"""One of the nineteen schemas of 42.2's bench that llguidance refuses by name rather than
compiling and ignoring — which is why it is the binding this stage chose."""


def checkpoint(hub: Path, model_id: str, vocab_size: int) -> None:
    """A catalog entry carrying the one field a mask reads. What makes the scan list a
    directory is a `config.json` with a `model_type` and the shard its index promises, and
    this is the whole of both."""
    head = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / "sha"
    head.mkdir(parents=True)
    (head / "config.json").write_text(json.dumps({"model_type": "test", "vocab_size": vocab_size}))
    # A shard with no tensors in it: eight bytes of header length and an empty header, which
    # is what admission sums when the same entry is read by an engine that has a store.
    (head / "model.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")
    revision = head.parents[1] / "refs" / "main"
    revision.parent.mkdir(parents=True)
    revision.write_text("sha")


def seed_settings(model_id: str, *, max_concurrent_requests: int) -> None:
    """One `model_settings` row before the engine reads it, the way `seed_config` seeds the
    config: the override is a row, and the engine reads rows synchronously."""
    migrate()
    with closing(sqlite3.connect(paths.server_db())) as connection, connection:
        connection.execute(
            "INSERT INTO model_settings(model, max_concurrent_requests) VALUES(?, ?)"
            " ON CONFLICT(model) DO UPDATE"
            " SET max_concurrent_requests = excluded.max_concurrent_requests",
            (model_id, max_concurrent_requests),
        )


async def piece(job: Job) -> Segment | None:
    """Bounded on purpose: a sentinel that stops being pushed has to fail the test, not hang
    the suite. There is no global pytest timeout, and the `engine.stop()` these tests put in
    a `finally` sits behind the drain, not in front of it."""
    return await asyncio.wait_for(job.chunks.get(), 30)


async def drain(job: Job) -> list[Segment]:
    pieces: list[Segment] = []
    while (chunk := await piece(job)) is not None:
        pieces.append(chunk)
    return pieces
