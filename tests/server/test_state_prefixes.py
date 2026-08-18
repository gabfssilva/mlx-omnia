"""The two prefix tiers `/admin/state` reports, and the button beside each of them.

Split off `test_state.py` for size alone: what is asserted here is the same route, read the
same way, over a store a request filled.
"""

import time
from pathlib import Path

from mlx_omnia import GenerationOptions, Text
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.db.models.prefixes import PrefixCacheFile
from mlx_omnia.server.runtime.engine import Engine
from tests.server.conftest import seed_config
from tests.server.state_stand import (
    WIDE_SPAN_BYTES,
    as_int,
    clear,
    drain,
    read,
    settled,
    tiny,
    wide,
    within,
)


def test_a_hot_store_is_live_memory_the_admission_reads() -> None:
    """A stored span outlives the request that closed it, so between requests it is live
    allocation and not pool — and `_measured` reads exactly that (`mx.get_active_memory`).
    Confirmed rather than read off the source: it is the figure admission evicts against, and
    a span invisible to it is a ceiling that lets a second model in on top of it.

    One process, one model, measured with the spans held and again after they are handed
    back. The store no longer dies with the model — a conversation survives an unload, which
    is the point of one store per daemon — so what releases them is the discard.
    """
    seed_config({"prefix_cache_bytes": 4 * WIDE_SPAN_BYTES, "prefix_span": 64})

    async def run() -> tuple[int, int]:
        engine = Engine(lambda _: wide(), Daemon())
        engine.start()
        try:
            # 64 is the narrowest span the config admits, and this tokenizer answers one id
            # whatever it is handed — so what crosses a boundary is the decode, not the
            # prompt.
            await drain(await engine.submit("w", Text("hi"), GenerationOptions(max_tokens=70)))
            held = settled()
            engine.discard_prefixes()
            assert await engine.unload("w")
            return held, settled()
        finally:
            await engine.stop()

    held, gone = within(run())
    assert held - gone > 0, "the spans were never live memory"


def test_the_memory_tier_is_reported_and_clearing_it_hands_it_back() -> None:
    """The two numbers the Server screen's prefix rows read, and the button beside them. The
    store is filled by a request, because that is the only thing that fills one."""
    seed_config({"prefix_cache_bytes": 4 * WIDE_SPAN_BYTES, "prefix_span": 64})

    async def run() -> tuple[int, int]:
        engine = Engine(lambda _: wide(), Daemon())
        engine.start()
        try:
            await drain(await engine.submit("w", Text("hi"), GenerationOptions(max_tokens=70)))
            held = await read(engine)
            await clear(engine, "memory")
            emptied = await read(engine)
            return as_int(held["prefix_memory_bytes"]), as_int(emptied["prefix_memory_bytes"])
        finally:
            await engine.stop()

    held, emptied = within(run())
    assert held > 0, "the spans the request stored were never reported"
    assert emptied == 0


def test_clearing_the_disk_tier_takes_the_rows_and_the_files(tmp_path: Path) -> None:
    """The floor that survives a restart, and the one thirty models pile up on. Nothing is
    generated here: what the route has to get right is that the row and the file leave
    together, and `prefixes.forget` is what it leans on to do it."""
    spilled = tmp_path / "w" / "cache.safetensors"
    spilled.parent.mkdir(parents=True)
    spilled.write_bytes(b"x" * 64)

    async def run() -> tuple[int, int, bool]:
        await PrefixCacheFile(
            key="k",
            kind="rows",
            model="w",
            path=str(spilled),
            bytes=64,
            created_at=time.time(),
            used_at=time.time(),
        ).save()
        engine = Engine(lambda _: tiny())
        held = await read(engine)
        await clear(engine, "disk")
        emptied = await read(engine)
        return (
            as_int(held["prefix_disk_bytes"]),
            as_int(emptied["prefix_disk_bytes"]),
            spilled.exists(),
        )

    held, emptied, left = within(run())
    assert (held, emptied) == (64, 0)
    assert not left, "the row went and the file stayed"
