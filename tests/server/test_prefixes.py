"""The cold floor under the prefix store: what is written, what comes back, and what the
index refuses.

The payloads here are built by hand rather than generated. What this file is about is the
bookkeeping — a key, an index, a ceiling and an LRU — and a real trunk would put a checkpoint
load in front of every assertion about a sqlite row. What the tensors are worth is
`test_cache_file.py`'s, and that the reuse produces the same ids is `test_generate.py`'s.

`FileVault`'s index is async and it is reached from whichever thread the generation runs on,
so the fixture below is the loop the daemon's lifespan would have bound: a real one, in a
thread of its own, with the database open on it.
"""

import asyncio
import threading
import time
from collections.abc import Coroutine, Iterator
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.cache_file import IDS, weight
from mlx_omnia.engine.core.prefix import Payload, Slot
from mlx_omnia.server.db import base as db
from mlx_omnia.server.db.models.prefixes import PrefixCacheFile
from mlx_omnia.server.main import migrate
from mlx_omnia.server.services import prefixes
from mlx_omnia.server.services.prefixes import FileVault

WIDTH = 64
"""Wide enough that one row is a kilobyte, so a span is a legible number of bytes."""


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "prefixes"
    monkeypatch.setattr(prefixes, "CACHE", directory)
    return directory


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """The loop a vault reaches its index through, in a thread of its own — the arrangement
    the daemon has, and the one `run_coroutine_threadsafe` needs."""
    migrate()
    running = asyncio.new_event_loop()
    thread = threading.Thread(target=running.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(db.connect(), running).result(10)
    yield running
    asyncio.run_coroutine_threadsafe(db.disconnect(), running).result(10)
    running.call_soon_threadsafe(running.stop)
    thread.join(5)


def awaited[T](loop: asyncio.AbstractEventLoop, work: Coroutine[object, object, T]) -> T:
    return asyncio.run_coroutine_threadsafe(work, loop).result(30)


def stored(loop: asyncio.AbstractEventLoop, model: str | None = None) -> list[PrefixCacheFile]:
    """The index, as the old `store.prefix_files` read it."""
    return awaited(loop, prefixes.rows(model))


def span(rows: int, tag: int = 0) -> Payload:
    """One span as `core.prefix` cuts it: named tensors and the ids they came from."""
    held: Payload = {
        "0.keys": mx.full((1, 2, rows, WIDTH), tag, dtype=mx.float32),
        "0.values": mx.full((1, 2, rows, WIDTH), tag, dtype=mx.float32),
        IDS: mx.arange(tag * rows, (tag + 1) * rows, dtype=mx.int32),
    }
    mx.eval(list(held.values()))
    return held


def vault(
    loop: asyncio.AbstractEventLoop, *, ceiling: int = 1 << 30, model: str = "m"
) -> FileVault:
    return FileVault(loop, model, ceiling=ceiling)


def written(tier: FileVault, key: Slot, payload: Payload) -> None:
    """A write and the wait for it: the tier writes behind the request on purpose, and a test
    that read straight back would be reading a queue."""
    tier.write(key, payload, weight(payload))
    tier.flush(10.0)


def describe_a_span_the_ceiling_pushed_out():
    def it_is_written_and_read_back(loop: asyncio.AbstractEventLoop):
        """The whole feature in one: what memory gave up on comes back off disk instead of
        being prefilled again."""
        tier = vault(loop)
        held = span(200)

        written(tier, ("rows", "abc"), held)
        read = tier.read(("rows", "abc"))

        assert read is not None
        assert mx.array_equal(read["0.keys"], held["0.keys"]).item()
        assert mx.array_equal(read[IDS], held[IDS]).item()

    def it_is_written_once(loop: asyncio.AbstractEventLoop):
        """Content-addressed and immutable: the same key is the same bytes, so a second write
        of it is a no-op and not a second file."""
        tier = vault(loop)

        written(tier, ("rows", "abc"), span(8))
        written(tier, ("rows", "abc"), span(8, tag=9))

        assert len(stored(loop, "m")) == 1

    def describe_when_the_daemon_starts_again():
        def it_finds_what_the_last_one_left(loop: asyncio.AbstractEventLoop):
            """Which is the point of a disk tier at all: the store dies with the process and
            the conversations that were warm in it are exactly the ones a restart loses."""
            written(vault(loop), ("rows", "abc"), span(16))

            reopened = vault(loop)

            assert reopened.holds(("rows", "abc"))
            assert reopened.read(("rows", "abc")) is not None

    def describe_when_the_file_is_gone():
        def it_is_a_miss_and_the_row_goes_with_it(loop: asyncio.AbstractEventLoop, cache: Path):
            """A truncated or vanished file is a miss and a prefill, which is always correct.
            The row goes too: an index pointing at nothing makes every lookup pay for the same
            failure, and the ceiling counts bytes nobody holds."""
            tier = vault(loop)
            written(tier, ("rows", "abc"), span(16))
            for file in cache.rglob("*.safetensors"):
                file.unlink()

            assert tier.read(("rows", "abc")) is None
            assert stored(loop, "m") == []


def describe_the_two_kinds_of_payload():
    def it_keeps_them_apart_under_one_key(loop: asyncio.AbstractEventLoop):
        """A span and the state that stood on its far boundary describe the same prefix and
        are kept under different rules, so they are two files and two rows."""
        tier = vault(loop)

        written(tier, ("rows", "abc"), span(8))
        written(tier, ("anchor", "abc"), span(4, tag=7))

        rows = tier.read(("rows", "abc"))
        anchor = tier.read(("anchor", "abc"))
        assert rows is not None and anchor is not None
        assert rows["0.keys"].shape[2] == 8 and anchor["0.keys"].shape[2] == 4
        assert {entry.kind for entry in stored(loop, "m")} == {"rows", "anchor"}


def describe_the_ceiling():
    def it_evicts_least_recently_used_first(loop: asyncio.AbstractEventLoop):
        one = span(8)
        tier = vault(loop, ceiling=weight(one) + weight(one) // 2)

        written(tier, ("rows", "one"), one)
        written(tier, ("rows", "two"), span(8, tag=1))

        assert [entry.key for entry in stored(loop, "m")] == ["two"]
        assert tier.read(("rows", "one")) is None

    def it_moves_a_row_on_a_hit(loop: asyncio.AbstractEventLoop):
        """The LRU is about use and not about age: a conversation somebody keeps coming back
        to is exactly the one worth the disk."""
        tier = vault(loop)
        written(tier, ("rows", "one"), span(8))
        written(tier, ("rows", "two"), span(8, tag=1))
        before = time.time()

        tier.read(("rows", "one"))

        touched = {entry.key: entry.used_at for entry in stored(loop, "m")}
        assert touched["one"] >= before
        assert touched["two"] < touched["one"]

    def describe_when_it_is_zero():
        def it_writes_nothing(loop: asyncio.AbstractEventLoop):
            tier = vault(loop, ceiling=0)

            written(tier, ("rows", "abc"), span(8))

            assert stored(loop, "m") == []


def describe_forgetting_a_model():
    def it_takes_the_rows_and_the_files(loop: asyncio.AbstractEventLoop, cache: Path):
        written(vault(loop), ("rows", "abc"), span(8))
        written(vault(loop, model="other"), ("rows", "def"), span(8))

        dropped = awaited(loop, prefixes.forget("m"))

        assert dropped == 1
        assert [entry.model for entry in stored(loop)] == ["other"]
        assert not (cache / "m").exists()


def describe_the_boot_sweep():
    def it_drops_rows_whose_file_is_gone(loop: asyncio.AbstractEventLoop, cache: Path):
        """A daemon killed between the rename and the insert leaves the opposite — a file
        with no row — which the ceiling would never count; that one is answered by the
        directory being one per model and disposable."""
        written(vault(loop), ("rows", "abc"), span(8))
        for file in cache.rglob("*.safetensors"):
            file.unlink()

        assert awaited(loop, prefixes.sweep()) == 1
        assert stored(loop) == []


def describe_a_burst_of_evictions():
    def it_lands_every_span_even_past_the_queue(loop: asyncio.AbstractEventLoop) -> None:
        # A drafted trunk commits a whole prompt's spans at once, and a conversation
        # bigger than the memory ceiling evicts hundreds of them in one loop — far past
        # the write-behind queue. Full is backpressure and never a drop: an overflowed
        # span that were silently discarded would leave the chain's head a hole, and
        # every resume of that conversation would prefill whole.
        slow = prefixes.dump

        def dragging(payload: Payload, path: Path) -> None:
            time.sleep(0.002)
            slow(payload, path)

        tier = vault(loop)
        original, prefixes.dump = prefixes.dump, dragging
        try:
            count = prefixes._QUEUE + 40
            for tag in range(count):
                accepted = tier.write(("rows", f"k{tag}"), span(4, tag), 4096)
                assert accepted
            tier.flush(30.0)
        finally:
            prefixes.dump = original
        assert sum(1 for _ in stored(loop, "m")) == count
