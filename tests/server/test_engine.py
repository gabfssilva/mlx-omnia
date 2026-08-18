import asyncio
import gc
import time
import weakref
from collections.abc import Iterator
from pathlib import Path

import pytest

from mlx_omnia import (
    Chat,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    Text,
    UnsupportedInput,
)
from mlx_omnia.engine.core.prefix import Prefixes
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime.engine import Engine, Job, NotResident
from mlx_omnia.server.services import catalog
from tests.server.conftest import engine_of, seed_config
from tests.server.engine_stand import FakeLanguageModel, caches_at, drain, piece


class SlowLanguageModel(FakeLanguageModel):
    """A stream long enough to walk away from, counting through the meter the way a real
    loop does — one piece, one id."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        self.calls.append((input, options))
        meter = options.meter
        if meter is not None:
            meter.prefill(len(input.read()))
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
            await engine.stop()

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
            await engine.stop()

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
            await engine.stop()

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
            await engine.stop()

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
            await engine.stop()

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
            await engine.stop()

    asyncio.run(run())


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
            await engine.stop()

    assert asyncio.run(run()), "the unloaded model stayed reachable from the worker's frame"


_BUDGET = 64 * 1024


def test_the_prefix_store_the_config_sizes_is_what_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How much of what they read the resident models may keep for the next request is
    `/admin/config`'s to say and the daemon's to hold: one store for every resident model,
    because which checkpoint a span came out of is inside its own key. What travels per model
    is that identity. An engine with no environment carries nothing — the cold path a daemon
    that was never wired runs on."""
    # Admission sizes the incoming model by scanning the two caches, so an engine that is
    # given an environment is an engine that scans: pointed at an empty directory rather than
    # at whatever this machine has downloaded.
    caches_at(monkeypatch, tmp_path)
    monkeypatch.setattr(catalog, "stamp_of", lambda model_id: f"{model_id}-stamp")
    seed_config({"prefix_cache_bytes": _BUDGET})

    async def run() -> tuple[Prefixes | None, Prefixes | None, Prefixes | None]:
        configured, second, plain = FakeLanguageModel(), FakeLanguageModel(), FakeLanguageModel()
        models = {"fake": configured, "other": second}
        with_store = engine_of(lambda name: CompositeModel(models[name], []))
        without = Engine(lambda _: CompositeModel(plain, []))
        with_store.start()
        without.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            await drain(await with_store.submit("fake", Text("hi"), asked))
            await drain(await with_store.submit("other", Text("hi"), asked))
            await drain(await without.submit("fake", Text("hi"), asked))
        finally:
            await with_store.stop()
            await without.stop()
        return tuple(model.calls[0][1].prefix for model in (configured, second, plain))

    first, other, plain = asyncio.run(run())
    assert first is not None and first.store.nbytes == 0
    # The ceiling is the machine's and not the model's: two resident models are weighed
    # against one another, which is only possible while it is literally one object.
    assert other is not None and other.store is first.store
    assert (first.model, other.model) == ("fake", "other"), "the key names the checkpoint"
    assert plain is None, "an engine with no environment keeps nothing"


def test_a_daemon_told_not_to_load_refuses_instead_of_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 3, and the whole of what `not_resident` buys: a cold load is seconds of the
    queue for everyone behind it, so a daemon configured to fail fast says so and never opens
    the checkpoint. `PUT /admin/models/{id}/residency` is an order rather than a request — so
    once the model is up, the same request goes through."""
    caches_at(monkeypatch, tmp_path)
    seed_config({"not_resident": "fail"})

    async def run() -> tuple[list[str], list[str], list[Segment]]:
        loads: list[str] = []

        def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
            loads.append(model_id)
            return CompositeModel(FakeLanguageModel(), [])

        engine = engine_of(loader)
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
            await engine.stop()
        return refused, loads, pieces

    refused, loaded, pieces = asyncio.run(run())
    assert refused == [], "the refusal has to come before the checkpoint is opened"
    assert loaded == ["fake"], "and the order that loaded it is what the next request used"
    assert pieces == [Segment("content", "hi")]
