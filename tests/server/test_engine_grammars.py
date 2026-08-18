"""The three lifetimes `constrain` decides, and the two refusals it owes a client."""

import asyncio
import gc
import weakref
from pathlib import Path

import pytest

from mlx_omnia import CompositeModel, GenerationOptions, Text
from mlx_omnia.engine.grammar import GrammarRefused
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime.engine import NotConstrainable, NotResident
from tests.server.conftest import engine_of, seed_config
from tests.server.engine_stand import (
    CITY,
    UNIQUE,
    ConstrainedLanguageModel,
    FakeLanguageModel,
    caches_at,
    checkpoint,
)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A catalog of this test's own. Never the machine's: a grammar reads the checkpoint's
    `config.json` through the same scan admission sizes with."""
    return caches_at(monkeypatch, tmp_path)


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
        engine = engine_of(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
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
            await engine.stop()

    asyncio.run(run())


def test_the_token_table_goes_away_with_the_model_it_was_built_for(hub: Path) -> None:
    """An unload gives the 150k entries back. The record is where they hang, so nothing has
    to remember to drop them — but nothing outside the record may hold them either, which is
    what a weak reference is here to say."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> bool:
        engine = engine_of(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
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
            await engine.stop()

    assert asyncio.run(run()), "the token table outlived the model it was built for"


def test_a_schema_the_compiler_will_not_take_is_refused_in_its_own_words(hub: Path) -> None:
    """The refusal is the product: a client that asked for a guarantee is owed the reason its
    schema is outside the subset, and `Unimplemented keys: [...]` is a reason where "grammar
    error" is not. Nothing is cached for it either — a refusal is not a compiled grammar."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> str:
        engine = engine_of(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
        engine.start()
        try:
            with pytest.raises(GrammarRefused) as refusal:
                await engine.constrain("test/plain", UNIQUE)
            assert engine.residency["test/plain"].grammars == {}
            return str(refusal.value)
        finally:
            await engine.stop()

    assert "uniqueItems" in asyncio.run(run())


def test_a_model_no_grammar_can_be_built_over_says_so_and_names_what_is_missing(
    hub: Path,
) -> None:
    """Not "model not available": the model is loaded and answering. What it has not got is a
    token table to compile against, and the way out belongs to the client — the same schema
    without `strict` is checked after the answer and needs none of it."""
    checkpoint(hub, "test/plain", 257)

    async def run() -> tuple[str, str]:
        tokenizerless = engine_of(lambda _: CompositeModel(FakeLanguageModel(), []))
        uncatalogued = engine_of(lambda _: CompositeModel(ConstrainedLanguageModel(), []))
        tokenizerless.start()
        uncatalogued.start()
        try:
            with pytest.raises(NotConstrainable) as blind:
                await tokenizerless.constrain("test/plain", CITY)
            with pytest.raises(NotConstrainable) as unlisted:
                await uncatalogued.constrain("test/elsewhere", CITY)
            return str(blind.value), str(unlisted.value)
        finally:
            await tokenizerless.stop()
            await uncatalogued.stop()

    blind, unlisted = asyncio.run(run())
    assert "tokenizer" in blind and "stop id" in blind
    assert "vocab_size" in unlisted


def test_a_daemon_told_not_to_load_does_not_load_for_a_grammar_either(hub: Path) -> None:
    """Decision 3 covers the compile too. A grammar needs the checkpoint's own token table,
    so building one for a model that is not resident is a cold load — seconds of the queue —
    bought by a request the same config would have refused a line later."""
    checkpoint(hub, "test/plain", 257)
    seed_config({"not_resident": "fail"})

    async def run() -> list[str]:
        loads: list[str] = []

        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return CompositeModel(ConstrainedLanguageModel(), [])

        engine = engine_of(loader)
        engine.start()
        try:
            with pytest.raises(NotResident):
                await engine.constrain("test/plain", CITY)
        finally:
            await engine.stop()
        return loads

    assert asyncio.run(run()) == [], "a grammar opened a checkpoint this daemon refuses to load"
