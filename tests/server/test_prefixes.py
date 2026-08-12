"""The disk tier under the prefix trie: what is written, what is read back, and what the
key refuses.

The caches here are built by hand rather than generated. What this file is about is the
bookkeeping — a key, an index, a ceiling and an LRU — and a real trunk would put a checkpoint
load in front of every assertion about a sqlite row. What the tensors are worth is
`test_cache_file.py`'s, and that the reuse produces the same ids is `test_generate.py`'s.
"""

import time
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.quant.quantization import Affine
from mlx_omnia.server import prefixes
from mlx_omnia.server.prefixes import DiskSpill
from mlx_omnia.server.store import Store

WIDTH = 64
"""Wide enough that one row is a kilobyte and a short cache clears the floor below without
the tests having to build a real conversation."""

FLOOR = 4096


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "prefixes"
    monkeypatch.setattr(prefixes, "CACHE", directory)
    return directory


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "server.db")


def filled(rows: int) -> list[LayerCache]:
    caches: list[LayerCache] = [KVCache(), DeltaCache()]
    block = mx.random.normal((1, 2, rows, WIDTH))
    attention = caches[0]
    assert isinstance(attention, KVCache)
    attention.update_and_fetch(block, block)
    recurrent = caches[1]
    assert isinstance(recurrent, DeltaCache)
    recurrent.offset = rows
    recurrent.state = mx.random.normal((1, 4, 8, 8))
    mx.eval([tensor for layer in caches for tensor in layer.tensors])
    return caches


def attending(rows: int) -> list[LayerCache]:
    """A trunk that rewinds, which is what a partial match needs — the mixed one above holds a
    recurrent state and has no way back to a branch point."""
    caches: list[LayerCache] = [KVCache(), KVCache()]
    block = mx.random.normal((1, 2, rows, WIDTH))
    for layer in caches:
        assert isinstance(layer, KVCache)
        layer.update_and_fetch(block, block)
    mx.eval([tensor for layer in caches for tensor in layer.tensors])
    return caches


def spill(
    store: Store, *, stamp: str = "sha", ceiling: int = 1 << 30, model: str = "m"
) -> DiskSpill:
    return DiskSpill(store, model, stamp=stamp, ceiling=ceiling, floor=FLOOR)


def test_an_evicted_prefix_is_written_and_read_back(store: Store) -> None:
    """The whole feature in one: what memory gave up on comes back off disk instead of being
    prefilled again."""
    tier = spill(store)
    tokens = list(range(200))

    tier.keep(tokens, filled(200), role="assistant")
    tier.flush()

    read = [KVCache(), DeltaCache()]
    covered = tier.recall([*tokens, 900, 901], read)
    assert covered == 200
    assert [layer.offset for layer in read] == [200, 200]
    assert len(store.prefix_files("m")) == 1


def test_a_prefix_that_covers_the_whole_prompt_is_not_offered(store: Store) -> None:
    """One row is always left to prefill: a forward needs one, and the logits of the last
    position are what the sampler reads. The same rule the trie enforces with `limit`."""
    tier = spill(store)
    tokens = list(range(200))
    tier.keep(tokens, filled(200), role="assistant")
    tier.flush()

    assert tier.recall(tokens, [KVCache(), DeltaCache()]) is None


def test_a_stored_cache_longer_than_the_match_is_rewound_to_it(store: Store) -> None:
    """What the ids in the index are for. A conversation whose last turn re-renders one token
    differently shares everything up to that token, and a lookup that could only answer "the
    same or nothing" would turn that into a whole prefill — which is the failure a restart
    makes permanent."""
    tier = spill(store)
    tokens = list(range(200))
    tier.keep(tokens, attending(200), role="assistant")
    tier.flush()
    edited = [*tokens[:120], 7777, *tokens[121:], 900]

    read: list[LayerCache] = [KVCache(), KVCache()]
    covered = tier.recall(edited, read)

    assert covered == 120
    assert [layer.offset for layer in read] == [120, 120]


def test_a_trunk_that_cannot_rewind_is_refused_a_partial_match(store: Store) -> None:
    """The other side of the same rule the trie follows: a recurrent state cannot be
    reconstructed backwards, so a file longer than the match is not a file this trunk can
    take. An exact prefix still is — that is the shape of a conversation."""
    tier = spill(store)
    tokens = list(range(200))
    tier.keep(tokens, filled(200), role="assistant")
    tier.flush()
    edited = [*tokens[:120], 7777, *tokens[121:], 900]

    assert tier.recall(edited, [DeltaCache(), DeltaCache()]) is None
    assert tier.recall([*tokens, 900], [KVCache(), DeltaCache()]) == 200


def test_a_short_prefix_is_not_worth_a_file(store: Store) -> None:
    """Below the floor the write costs more than the prefill it saves, and every one of them
    is a row that every lookup walks."""
    tier = spill(store)

    tier.keep([1, 2, 3], filled(1), role="assistant")
    tier.flush()

    assert store.prefix_files("m") == []


def test_a_checkpoint_that_moved_under_the_id_is_a_miss(store: Store) -> None:
    """The failure the stamp is in the key for, and the one a restart makes permanent: the
    same id, different weights, and a cache that would decode fluently off the wrong ones."""
    tokens = list(range(200))
    written = spill(store)
    written.keep(tokens, filled(200), role="assistant")
    written.flush()

    moved = spill(store, stamp="another")

    assert moved.recall([*tokens, 900], [KVCache(), DeltaCache()]) is None


def test_a_different_kv_policy_is_a_miss(store: Store) -> None:
    """A dense cache and a compressed one are two formats of the same thing, and only one of
    them is what the trunk about to read it builds. The policy is read off the trunk itself
    and never off a caller's configuration: the same `DiskSpill` serves whatever the model
    made, so what has to differ here is the cache and not the tier."""
    tokens = list(range(200))
    tier = spill(store)
    tier.keep(tokens, filled(200), role="assistant")
    tier.flush()

    compressed: list[LayerCache] = [
        QuantizedKVCache(Affine(group_size=64, bits=4), Affine(group_size=64, bits=4)),
        DeltaCache(),
    ]
    assert tier.recall([*tokens, 900], compressed) is None
    # And the same tier still hits for the trunk that wrote it, so the miss above is the
    # policy and not the tier having stopped working.
    assert tier.recall([*tokens, 900], [KVCache(), DeltaCache()]) == 200


def test_a_corrupted_file_is_a_miss_and_its_row_goes_with_it(store: Store) -> None:
    """A truncated file is a prefill, which is always correct. The row is dropped too: an
    index pointing at nothing would make every lookup pay for the same failure, and the
    ceiling would be enforced against bytes nobody holds."""
    tier = spill(store)
    tokens = list(range(200))
    tier.keep(tokens, filled(200), role="assistant")
    tier.flush()
    (entry,) = store.prefix_files("m")
    whole = Path(entry.path).read_bytes()
    Path(entry.path).write_bytes(whole[: len(whole) // 2])

    assert tier.recall([*tokens, 900], [KVCache(), DeltaCache()]) is None
    assert store.prefix_files("m") == []


def test_a_write_that_failed_leaves_no_row_and_no_file(
    store: Store, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk, a directory that cannot be made, a process killed between the staging
    file and the rename. The row is written after the rename and never before, so what a
    failure leaves is the state the daemon was in before the feature — and nothing will look
    for a file that is not there."""

    def refuses(*_: object, **__: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(prefixes, "dump", refuses)
    tier = spill(store)

    tier.keep(list(range(200)), filled(200), role="assistant")
    tier.flush()

    assert store.prefix_files("m") == []
    assert not any(cache.rglob("*.safetensors"))


def test_the_ceiling_evicts_the_least_recently_used(store: Store) -> None:
    """Without a ceiling this fills the disk in an afternoon. By last use and not by age: a
    conversation somebody keeps coming back to is the one worth the space."""
    tier = spill(store)
    first, second = list(range(200)), list(range(1000, 1200))
    tier.keep(first, filled(200), role="assistant")
    tier.flush()
    tier.keep(second, filled(200), role="assistant")
    tier.flush()
    # A hit on the first one makes it the recent one, which is what the LRU has to read.
    assert tier.recall([*first, 900], [KVCache(), DeltaCache()]) == 200
    held = store.prefix_bytes()

    tight = spill(store, ceiling=held - 1)
    tight.keep(list(range(2000, 2200)), filled(200), role="assistant")
    tight.flush()

    kept = {entry.tokens for entry in store.prefix_files("m")}
    assert store.prefix_bytes() <= held - 1
    assert kept, "the ceiling emptied the cache instead of trimming it"
    assert tight.recall([*second, 900], [KVCache(), DeltaCache()]) is None, "the stale one went"


def test_a_hit_moves_the_last_use(store: Store) -> None:
    tier = spill(store)
    tokens = list(range(200))
    tier.keep(tokens, filled(200), role="assistant")
    tier.flush()
    (before,) = store.prefix_files("m")
    time.sleep(0.01)

    tier.recall([*tokens, 900], [KVCache(), DeltaCache()])

    (after,) = store.prefix_files("m")
    assert after.used_at > before.used_at


def test_a_ceiling_of_zero_writes_nothing_and_reads_nothing(store: Store) -> None:
    """How the disk tier is turned off, and it has to be off on both sides: a daemon that
    stopped writing but went on reading would serve caches the user asked it to forget."""
    tier = spill(store, ceiling=0)

    tier.keep(list(range(200)), filled(200), role="assistant")
    tier.flush()

    assert store.prefix_files("m") == []
    assert tier.recall(list(range(201)), [KVCache(), DeltaCache()]) is None


def test_forgetting_a_model_takes_its_rows_and_its_files(store: Store, cache: Path) -> None:
    """What a checkpoint's deletion must do. Not what an unload does — surviving an unload is
    the whole point — but nothing keyed to a checkpoint outlives it."""
    keeper = spill(store, model="one")
    keeper.keep(list(range(200)), filled(200), role="assistant")
    keeper.flush()
    other = spill(store, model="two")
    other.keep(list(range(300, 500)), filled(200), role="assistant")
    other.flush()

    dropped = prefixes.forget(store, "one")

    assert dropped == 1
    assert store.prefix_files("one") == []
    assert len(store.prefix_files("two")) == 1
    assert not prefixes.directory("one").exists()


def test_a_row_whose_file_is_gone_is_swept_at_boot(store: Store) -> None:
    """A cache directory somebody cleared under a database that was not. Left in, the ceiling
    would be enforced against bytes nobody holds."""
    tier = spill(store)
    tier.keep(list(range(200)), filled(200), role="assistant")
    tier.flush()
    (entry,) = store.prefix_files("m")
    Path(entry.path).unlink()

    assert prefixes.sweep(store) == 1
    assert store.prefix_files() == []


def test_the_directory_is_private_to_the_user(store: Store) -> None:
    """The first state the daemon writes that is derived from what the user said to a model.
    Every other cache it keeps holds weights, which are already on disk in the open."""
    tier = spill(store)
    tier.keep(list(range(200)), filled(200), role="assistant")
    tier.flush()

    assert prefixes.directory("m").stat().st_mode & 0o777 == 0o700
