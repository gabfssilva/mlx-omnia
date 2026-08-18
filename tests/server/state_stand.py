"""The stand `/admin/state` is read over: the fake models whose bytes are known, the mount
that answers the route, and the loop the database is opened on.

Shared rather than duplicated because the suite is split by what it is about — what is
resident and what it costs on one side, the two prefix tiers on the other — and both halves
read the same route over the same fakes.
"""

import asyncio
import gc
import time
from collections.abc import Coroutine, Iterator, Sequence
from typing import TypeIs

import httpx
import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI

from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    KVCache,
    ModelInput,
    ModelSignature,
    Text,
    TextLanguageModel,
)
from mlx_omnia.engine.core.attend import attend
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.api.management.state import router
from mlx_omnia.server.db import base as db
from mlx_omnia.server.main import migrate
from mlx_omnia.server.runtime.engine import Engine, Job

KV_BYTES = 32 * 1024 * 1024

SPIKE_BYTES = 256 * 1024 * 1024
"""Larger than any figure a correct KV reading can produce, and released before the request
that is measured: it is the peak that must not survive into the next request's number."""

TINY_BYTES = 8 * 4 * 4
"""The embedding below and nothing else: 8 rows of 4 float32."""


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return self.embed(ids)


_WIDE = 16 * 1024
"""Columns per cached row. The KV buffer grows in blocks of 256 rows, so one prompt reserves
`256 * _WIDE * 4` bytes twice over — 32 MiB, which is a figure that stands clear of whatever
the allocator is doing around it."""

WIDE_SPAN_BYTES = 2 * 256 * _WIDE * 4
"""What one span of this cache weighs: keys and values, 256 rows, `_WIDE` wide, in fp32 —
32 MiB, a figure that stands clear of whatever the allocator is doing around it."""


class WideLM(nn.Module):
    """`TinyLM` with a cache worth measuring: what the store keeps between requests are these
    rows, and at four kilobytes the question of whether anything sees them cannot be
    asked."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        out = self.embed(ids)
        if cache is None:
            return out
        rows = mx.zeros((ids.shape[0], 1, ids.shape[1], _WIDE), dtype=mx.float32)
        # Through the door a trunk uses, so the same fake serves a lone cache and a ragged
        # batch's adapter. The head reads the result back, and that is what makes the rows
        # exist: mlx is lazy, so a fake whose output ignores its own cache leaves the write
        # unevaluated — and memory nothing allocated is memory nothing measures.
        read = attend(cache[0], rows, keys=rows, values=rows, scale=1.0, mask=None)
        return out + read[0, 0, -1, 0]


def wide() -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(TextLanguageModel(WideLM(), TinyTokenizer()), [])


class TinyTokenizer:
    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        return iter([0])

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"."


def tiny() -> CompositeModel[Text, Segment, GenerationOptions]:
    """The production chain down to the tree: the engine holds a `CompositeModel`, which
    holds the task-level model, which holds the `nn.Module` the bytes come off."""
    return CompositeModel(TextLanguageModel(TinyLM(), TinyTokenizer()), [])


class HoldingLanguageModel:
    """Holds an `mx.array` for as long as its stream runs, which is what a KV cache is to
    the accounting: memory the request takes on top of the settled weights and gives back
    when it ends."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        cache = mx.zeros((KV_BYTES // 4,), dtype=mx.float32)
        mx.eval(cache)
        for index in range(options.max_tokens):
            time.sleep(self.delay)
            yield Segment("content", str(index))


def holding(delay: float = 0.0) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(HoldingLanguageModel(delay), [])


BIG_BYTES = 128 * 1024 * 1024

BIG_ROWS = BIG_BYTES // (4 * 4)


class BigLM(TinyLM):
    """Weights large enough that charging them to another model's KV is unmistakable."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(BIG_ROWS, 4)


def big() -> CompositeModel[Text, Segment, GenerationOptions]:
    model = BigLM()
    # The load is what allocates: a lazy tree costs nothing, and the point of the test is
    # the memory landing while another model is decoding.
    mx.eval(model.parameters())
    return CompositeModel(TextLanguageModel(model, TinyTokenizer()), [])


def settled() -> int:
    """What mlx holds with nothing in flight: the collector first, because a buffer a dead
    Python object still owns is not the allocator's fault, and `clear_cache` after it, so the
    pool's free blocks are not counted as somebody's."""
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


async def piece(job: Job) -> Segment | None:
    """Bounded on purpose: a sentinel that stops being pushed has to fail the test, not
    hang the suite — there is no global pytest timeout."""
    return await asyncio.wait_for(job.chunks.get(), 30)


async def drain(job: Job) -> None:
    while await piece(job) is not None:
        pass


def mounted(engine: Engine) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine
    app.include_router(router)
    return app


async def read(engine: Engine) -> dict[str, object]:
    transport = httpx.ASGITransport(app=mounted(engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://state") as client:
        response = await asyncio.wait_for(client.get("/admin/state"), 30)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


async def clear(engine: Engine, tier: str) -> None:
    transport = httpx.ASGITransport(app=mounted(engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://state") as client:
        response = await asyncio.wait_for(client.delete(f"/admin/prefixes/{tier}"), 30)
    assert response.status_code == 204, response.text


def within[T](work: Coroutine[object, object, T]) -> T:
    """The route reports what the spilled conversations weigh, which is a row count and lives
    in the database — so every read here runs on a loop with it open, the way the lifespan
    would have opened it."""

    async def main() -> T:
        await db.connect()
        try:
            return await work
        finally:
            await db.disconnect()

    migrate()
    return asyncio.run(main())


def models_of(body: dict[str, object]) -> list[dict[str, object]]:
    models = body["models"]
    assert isinstance(models, list)
    for entry in models:
        assert isinstance(entry, dict)
    return models


def as_int(value: object) -> int:
    assert isinstance(value, int)
    return value
