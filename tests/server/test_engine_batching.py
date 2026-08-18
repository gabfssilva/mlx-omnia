"""What compatible requests share, and what the concurrency limits keep apart."""

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia import CompositeModel, GenerationOptions, Text, TextLanguageModel
from mlx_omnia.engine.batching import BatchedKVCache
from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.qwen3.config import Qwen3Config
from mlx_omnia.engine.models.qwen3.model import Qwen3
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime.engine import Job
from tests.server.conftest import engine_of, seed_config
from tests.server.engine_stand import AsciiTokenizer, caches_at, drain, piece, seed_settings


class BatchCountingTrunk:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.batched_steps = 0

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[KVStore]) -> mx.array:
        self.batch_sizes.append(ids.shape[0])
        if isinstance(cache[0], BatchedKVCache):
            self.batched_steps += 1
        return -mx.abs(mx.arange(128) - (ids + 1)[..., None]).astype(mx.float32)


class BlockingBatchTrunk:
    def __init__(self) -> None:
        self.inner: Qwen3 | None = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def _model(self) -> Qwen3:
        if self.inner is None:
            self.inner = Qwen3(
                Qwen3Config(
                    hidden_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    vocab_size=128,
                    rms_norm_eps=1e-6,
                    rope_theta=10_000,
                    intermediate_size=32,
                )
            )
        return self.inner

    def make_cache(self) -> list[KVCache]:
        return self._model().make_cache()

    def __call__(self, ids: mx.array, cache: Sequence[KVStore]) -> mx.array:
        if ids.shape[0] == 2:
            self.entered.set()
            assert self.release.wait(5), "test did not release the batched step"
        return self._model()(ids, cache)


@pytest.fixture(autouse=True)
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    caches_at(monkeypatch, tmp_path)


def test_compatible_requests_share_decode_forwards() -> None:
    seed_config({"max_concurrent_requests": 2, "prefix_cache_bytes": 0})

    async def run() -> tuple[list[Segment], list[Segment], list[int]]:
        trunk = BatchCountingTrunk()
        engine = engine_of(lambda _: CompositeModel(TextLanguageModel(trunk, AsciiTokenizer()), []))
        engine.start()
        try:
            first, second = await asyncio.gather(
                engine.submit("fake", Text("A"), GenerationOptions(max_tokens=2)),
                engine.submit("fake", Text("E"), GenerationOptions(max_tokens=2)),
            )
            one, two = await asyncio.gather(drain(first), drain(second))
            return one, two, trunk.batch_sizes
        finally:
            await engine.stop()

    first, second, batches = asyncio.run(run())
    assert first == [Segment("content", "B"), Segment("content", "C")]
    assert second == [Segment("content", "F"), Segment("content", "G")]
    assert batches == [1, 1, 2, 2]


def test_a_request_joins_a_batch_that_is_already_decoding() -> None:
    seed_config({"max_concurrent_requests": 2, "prefix_cache_bytes": 0})

    async def run() -> tuple[list[Segment], list[Segment], list[int]]:
        trunk = BatchCountingTrunk()
        engine = engine_of(lambda _: CompositeModel(TextLanguageModel(trunk, AsciiTokenizer()), []))
        engine.start()
        try:
            first = await engine.submit("fake", Text("A"), GenerationOptions(max_tokens=4))
            head = await piece(first)
            second = await engine.submit("fake", Text("E"), GenerationOptions(max_tokens=2))
            one, two = await asyncio.gather(drain(first), drain(second))
            return ([] if head is None else [head, *one]), two, trunk.batch_sizes
        finally:
            await engine.stop()

    first, second, batches = asyncio.run(run())
    assert first == [Segment("content", letter) for letter in "BCDE"]
    assert second == [Segment("content", letter) for letter in "FG"]
    assert 2 in batches


def test_cancelling_one_batched_request_does_not_change_the_other() -> None:
    seed_config({"max_concurrent_requests": 2, "prefix_cache_bytes": 0})

    async def run() -> tuple[Job, list[Segment]]:
        trunk = BatchCountingTrunk()
        engine = engine_of(lambda _: CompositeModel(TextLanguageModel(trunk, AsciiTokenizer()), []))
        engine.start()
        try:
            cancelled, survivor = await asyncio.gather(
                engine.submit("fake", Text("A"), GenerationOptions(max_tokens=8)),
                engine.submit("fake", Text("E"), GenerationOptions(max_tokens=4)),
            )
            assert await piece(cancelled) == Segment("content", "B")
            cancelled.cancel()
            await drain(cancelled)
            return cancelled, await drain(survivor)
        finally:
            await engine.stop()

    cancelled, survivor = asyncio.run(run())
    assert cancelled.state == "cancelled"
    assert survivor == [Segment("content", letter) for letter in "FGHI"]


def test_the_model_concurrency_override_caps_the_global_limit() -> None:
    seed_config({"max_concurrent_requests": 2, "prefix_cache_bytes": 0})
    seed_settings("fake", max_concurrent_requests=1)

    async def run() -> tuple[list[int], int]:
        trunk = BatchCountingTrunk()
        engine = engine_of(lambda _: CompositeModel(TextLanguageModel(trunk, AsciiTokenizer()), []))
        engine.start()
        try:
            first, second = await asyncio.gather(
                engine.submit("fake", Text("A"), GenerationOptions(max_tokens=2)),
                engine.submit("fake", Text("E"), GenerationOptions(max_tokens=2)),
            )
            await asyncio.gather(drain(first), drain(second))
            return trunk.batch_sizes, trunk.batched_steps
        finally:
            await engine.stop()

    sizes, _ = asyncio.run(run())
    # Every forward carries one row: the override kept the two requests out of each
    # other's ticks. `batched_steps` stopped being zero when `stream()` itself became a
    # batch of 1 — the machinery is shared now, and what the override governs is the
    # grouping, not the machinery.
    assert sizes == [1, 1, 1, 1, 1, 1]


def test_running_and_kv_bytes_report_the_active_batch() -> None:
    seed_config({"max_concurrent_requests": 2, "prefix_cache_bytes": 0})

    async def run() -> tuple[int, int]:
        trunk = BlockingBatchTrunk()
        engine = engine_of(lambda _: CompositeModel(TextLanguageModel(trunk, AsciiTokenizer()), []))
        try:
            first, second = await asyncio.gather(
                engine.submit("fake", Text("A"), GenerationOptions(max_tokens=1)),
                engine.submit("fake", Text("E"), GenerationOptions(max_tokens=1)),
            )
            engine.start()
            assert await asyncio.to_thread(trunk.entered.wait, 5), (
                first.state,
                first.error,
                second.state,
                second.error,
            )
            running = engine.running
            kv_bytes = engine.residency["fake"].kv_bytes
            trunk.release.set()
            await asyncio.gather(drain(first), drain(second))
            return running, kv_bytes
        finally:
            trunk.release.set()
            await engine.stop()

    assert asyncio.run(run()) == (2, 65_536)
