import asyncio
import gc
import json
import time
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import TypeIs

import pytest

from sideros import (
    TEXT,
    Chat,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
    UnsupportedInput,
)
from sideros.grammar import GrammarRefused
from sideros.suppress import Segment
from sideros_server import Engine, catalog
from sideros_server.engine import Job, NotConstrainable, NotResident
from sideros_server.store import Store


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


class SlowLanguageModel(FakeLanguageModel):
    """A stream long enough to walk away from, counting through the meter the way a real
    loop does — one piece, one id."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        self.calls.append((input, options))
        meter = options.meter
        if meter is not None:
            meter.prefill(len(input.value))
        for index in range(options.max_tokens):
            time.sleep(0.01)
            if meter is not None:
                meter.token()
            yield Segment("content", str(index))


class FailingLanguageModel(FakeLanguageModel):
    """Fails mid-generation, which is the only way the worker meets an exception: an input
    the model refuses outright never becomes a job."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        self.calls.append((input, options))
        yield Segment("content", "half a ")
        raise RuntimeError("the checkpoint went away")


def require_language_model(model: LanguageModel[Text]) -> None:
    pass


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


def test_engine_uses_language_model_and_generation_options() -> None:
    async def run() -> tuple[list[Segment], FakeLanguageModel]:
        model = FakeLanguageModel()
        require_language_model(model)
        engine = Engine(lambda _: CompositeModel(model, []))
        engine.start()
        try:
            job = await engine.submit("fake", Text("hello"), GenerationOptions(max_tokens=3))
            return await drain(job), model
        finally:
            engine.stop()

    chunks, model = asyncio.run(run())
    assert chunks == [Segment("content", "hel")]
    assert model.calls == [(Text("hello"), GenerationOptions(max_tokens=3))]


def test_the_engine_hands_the_job_s_meter_to_the_model() -> None:
    """The counts exist on the far side of `stream` — the conversation is rendered by the
    checkpoint's own template and tokenized inside the model — so the meter travels in
    with the options rather than the numbers travelling out."""

    async def run() -> tuple[Job, FakeLanguageModel]:
        model = FakeLanguageModel()
        engine = Engine(lambda _: CompositeModel(model, []))
        engine.start()
        try:
            job = await engine.submit("fake", Text("hello"), GenerationOptions(max_tokens=3))
            await drain(job)
            return job, model
        finally:
            engine.stop()

    job, model = asyncio.run(run())
    assert model.calls[0][1].meter is job.meter


def test_a_job_that_runs_to_the_end_is_completed() -> None:
    """And stays completed: every response ends in a `cancel()` — it is the `finally` of
    the SSE generator — which must not rewrite a job that already finished."""

    async def run() -> Job:
        engine = Engine(lambda _: CompositeModel(FakeLanguageModel(), []))
        engine.start()
        try:
            job = await engine.submit("fake", Text("hello"), GenerationOptions(max_tokens=3))
            await drain(job)
            job.cancel()
            return job
        finally:
            engine.stop()

    assert asyncio.run(run()).state == "completed"


def test_a_client_that_walks_away_leaves_a_cancelled_job() -> None:
    async def run() -> Job:
        engine = Engine(lambda _: CompositeModel(SlowLanguageModel(), []))
        engine.start()
        try:
            job = await engine.submit("fake", Text("hello"), GenerationOptions(max_tokens=64))
            for _ in range(2):
                assert await piece(job) is not None
            job.cancel()
            await drain(job)
            return job
        finally:
            engine.stop()

    job = asyncio.run(run())
    assert job.state == "cancelled"
    assert 0 < job.meter.completion_tokens < 64


def test_a_model_that_fails_mid_generation_leaves_an_error_job() -> None:
    async def run() -> Job:
        engine = Engine(lambda _: CompositeModel(FailingLanguageModel(), []))
        engine.start()
        try:
            job = await engine.submit("fake", Text("hello"), GenerationOptions(max_tokens=3))
            await drain(job)
            return job
        finally:
            engine.stop()

    job = asyncio.run(run())
    assert job.state == "error"
    assert job.error is not None
    assert "RuntimeError" in job.error


def test_an_input_the_model_refuses_never_becomes_a_job() -> None:
    """The queue is serial: a job that would fail mid-generation blocks the ones behind it
    for as long as the failure takes. The refusal happens before it is queued."""

    async def run() -> None:
        engine = Engine(lambda _: CompositeModel(FakeLanguageModel(), []))
        engine.start()
        try:
            with pytest.raises(UnsupportedInput):
                await engine.submit(
                    "fake",
                    Chat(({"role": "user", "content": "hi"},)),
                    GenerationOptions(max_tokens=3),
                )
        finally:
            engine.stop()

    asyncio.run(run())


_BUDGET = 64 * 1024


def test_the_prefix_budget_the_config_holds_is_what_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How much of what it reads a model may keep for the next request is `/admin/config`'s
    to say and the model's to hold: the engine carries the number and nothing else, because
    the cache's element type is the trunk's own and a `LanguageModel[ModelInput]` cannot name
    it. An engine with no store carries 0 — the cold path every other test here runs on."""
    # Admission sizes the incoming model by scanning the two caches, so an engine that is
    # given a store is an engine that scans: pointed at an empty directory rather than at
    # whatever this machine has downloaded.
    monkeypatch.setattr(catalog, "HUB_CACHE", tmp_path / "hub")
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")

    async def run() -> tuple[int, int]:
        store = Store(tmp_path / "server.db")
        store.set_config({"prefix_cache_bytes": json.dumps(_BUDGET)})
        configured, plain = FakeLanguageModel(), FakeLanguageModel()
        with_store = Engine(lambda _: CompositeModel(configured, []), store)
        without = Engine(lambda _: CompositeModel(plain, []))
        with_store.start()
        without.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            await drain(await with_store.submit("fake", Text("hi"), asked))
            await drain(await without.submit("fake", Text("hi"), asked))
        finally:
            with_store.stop()
            without.stop()
        return configured.calls[0][1].prefix_budget, plain.calls[0][1].prefix_budget

    assert asyncio.run(run()) == (_BUDGET, 0)


_STOP = 256
_PIECES = {value: bytes([value]) for value in range(256)} | {_STOP: b"<|end|>"}


class ByteTokenizer:
    """One id per byte plus one that ends a turn. Small enough to build a table over in a
    test and real enough for a grammar: JSON is bytes, and llguidance walks these."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

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


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A catalog of this test's own. Never the machine's: a grammar reads the checkpoint's
    `config.json` through the same scan admission sizes with."""
    monkeypatch.setattr(catalog, "HUB_CACHE", tmp_path / "hub")
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    return tmp_path / "hub"


def test_the_token_table_is_per_resident_model_and_the_walk_is_per_request(hub: Path) -> None:
    """The three lifetimes, which is the whole of what `constrain` decides.

    The table is the expensive one — 0.27 s and 150k entries over a real vocabulary — and it
    is per *model*: keyed by schema instead, two residents asked for the same schema would
    share one table, and the second model's mask would cover the first model's ids. Its width
    says which is which here, and it comes off `config.json`: `test/padded` declares a head
    wider than its tokenizer knows, the way Qwen3's 151936 sits over 151669 ids.

    The grammar between them is per (model, schema) and shared. The walk never is: everything
    a request changes lives in it.
    """
    checkpoint(hub, "test/plain", 257)
    checkpoint(hub, "test/padded", 300)

    async def run() -> None:
        engine = Engine(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
        engine.start()
        try:
            first = await engine.constrain("test/plain", CITY)
            other = await engine.constrain("test/padded", CITY)
            again = await engine.constrain("test/plain", CITY)
            plain = engine.residency["test/plain"].vocabulary
            padded = engine.residency["test/padded"].vocabulary
            assert plain is not None and padded is not None
            assert plain is not padded, "two resident models shared one token table"
            assert (plain.size, padded.size) == (257, 300), "the head's width, not the count"
            assert first is not other and first is not again, "two requests shared one walk"
            assert len(engine.residency["test/plain"].grammars) == 1, "one schema, one grammar"
        finally:
            engine.stop()

    asyncio.run(run())


def test_the_token_table_goes_away_with_the_model_it_was_built_for(hub: Path) -> None:
    """An unload gives the 150k entries back. The record is where they hang, so nothing has
    to remember to drop them — but nothing outside the record may hold them either, which is
    what a weak reference is here to say."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> bool:
        engine = Engine(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
        engine.start()
        try:
            await engine.constrain("test/plain", CITY)
            built = engine.residency["test/plain"].vocabulary
            assert built is not None
            table = weakref.ref(built)
            # The one strong reference this test itself holds. Kept out of the way rather
            # than left to the frame: it would answer the question on its own.
            del built
            assert await engine.unload("test/plain")
            gc.collect()
            return table() is None
        finally:
            engine.stop()

    assert asyncio.run(run()), "the token table outlived the model it was built for"


def test_an_unload_lets_go_of_the_finished_job_s_model() -> None:
    """The worker suspends on the queue with its loop frame alive, and a local still naming
    the finished job keeps its model reachable — the `_Release` an unload sends down the
    queue rebinds `item`, never `job`. On a 30B that local is 19 GB the 204 said came back."""

    async def run() -> bool:
        engine = Engine(lambda _: CompositeModel(FakeLanguageModel(), []))
        engine.start()
        try:
            job = await engine.submit("fake", Text("hello"), GenerationOptions(max_tokens=3))
            await drain(job)
            model = weakref.ref(job.model)
            # The one strong reference this test itself holds. Kept out of the way rather
            # than left to the frame: it would answer the question on its own.
            del job
            assert await engine.unload("fake")
            gc.collect()
            return model() is None
        finally:
            engine.stop()

    assert asyncio.run(run()), "the unloaded model stayed reachable from the worker's frame"


def test_a_schema_the_compiler_will_not_take_is_refused_in_its_own_words(hub: Path) -> None:
    """The refusal is the product: a client that asked for a guarantee is owed the reason its
    schema is outside the subset, and `Unimplemented keys: [...]` is a reason where "grammar
    error" is not. Nothing is cached for it either — a refusal is not a compiled grammar."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> str:
        engine = Engine(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
        engine.start()
        try:
            with pytest.raises(GrammarRefused) as refusal:
                await engine.constrain("test/plain", UNIQUE)
            assert engine.residency["test/plain"].grammars == {}
            return str(refusal.value)
        finally:
            engine.stop()

    assert "uniqueItems" in asyncio.run(run())


def test_a_model_no_grammar_can_be_built_over_says_so_and_names_what_is_missing(
    hub: Path,
) -> None:
    """Not "model not available": the model is loaded and answering. What it has not got is a
    token table to compile against, and the way out belongs to the client — the same schema
    without `strict` is checked after the answer and needs none of it."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> tuple[str, str]:
        tokenizerless = Engine(lambda _: CompositeModel(FakeLanguageModel(), []))
        uncatalogued = Engine(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
        tokenizerless.start()
        uncatalogued.start()
        try:
            with pytest.raises(NotConstrainable) as blind:
                await tokenizerless.constrain("test/plain", CITY)
            with pytest.raises(NotConstrainable) as unlisted:
                await uncatalogued.constrain("test/elsewhere", CITY)
            return str(blind.value), str(unlisted.value)
        finally:
            tokenizerless.stop()
            uncatalogued.stop()

    blind, unlisted = asyncio.run(run())
    assert "tokenizer" in blind and "stop id" in blind
    assert "vocab_size" in unlisted


def test_a_daemon_told_not_to_load_does_not_load_for_a_grammar_either(hub: Path) -> None:
    """Decision 3 covers the compile too. A grammar needs the checkpoint's own token table,
    so building one for a model that is not resident is a cold load — seconds of the queue —
    bought by a request the same config would have refused a line later."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> list[str]:
        store = Store(hub.parent / "server.db")
        store.set_config({"not_resident": json.dumps("fail")})
        loads: list[str] = []

        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return CompositeModel(ConstrainedLanguageModel(), [])

        engine = Engine(loader, store)
        engine.start()
        try:
            with pytest.raises(NotResident):
                await engine.constrain("test/plain", CITY)
        finally:
            engine.stop()
        return loads

    assert asyncio.run(run()) == [], "a grammar opened a checkpoint this daemon refuses to load"


def test_a_daemon_told_not_to_load_refuses_instead_of_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 3, and the whole of what `not_resident` buys: a cold load is seconds of the
    queue for everyone behind it, so a daemon configured to fail fast says so and never opens
    the checkpoint. `PUT /admin/models/{id}/residency` is an order rather than a request — so
    once the model is up, the same request goes through."""
    monkeypatch.setattr(catalog, "HUB_CACHE", tmp_path / "hub")
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")

    async def run() -> tuple[list[str], list[str], list[Segment]]:
        store = Store(tmp_path / "server.db")
        store.set_config({"not_resident": json.dumps("fail")})
        loads: list[str] = []

        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return CompositeModel(FakeLanguageModel(), [])

        engine = Engine(loader, store)
        engine.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            with pytest.raises(NotResident) as refusal:
                await engine.submit("fake", Text("hi"), asked)
            assert "fake" in str(refusal.value), "a refusal that does not name the model"
            refused = list(loads)
            await engine.resolve("fake")
            pieces = await drain(await engine.submit("fake", Text("hi"), asked))
        finally:
            engine.stop()
        return refused, loads, pieces

    refused, loaded, pieces = asyncio.run(run())
    assert refused == [], "the refusal has to come before the checkpoint is opened"
    assert loaded == ["fake"], "and the order that loaded it is what the next request used"
    assert pieces == [Segment("content", "hi")]
