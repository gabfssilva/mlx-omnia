"""A trunk's cache as a file: what it writes, what it refuses, and what comes back.

Every equality here is an equality and not a floor. Nothing in this path does arithmetic —
tensors go out and the same tensors come in — so a tolerance would only be hiding a bug in
whichever of the two directions rounded.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia.engine.core import cache_file
from mlx_omnia.engine.core.cache import (
    ConvCache,
    DeltaCache,
    FixedKVCache,
    KVCache,
    LayerCache,
    RingKVCache,
    SharedKVReader,
)


def filled(rows: int, width: int = 8) -> list[LayerCache]:
    """The `bailing_hybrid` shape in miniature: attention beside recurrence, plus a layer
    that borrows another's rows."""
    caches: list[LayerCache] = [KVCache(), DeltaCache(), SharedKVReader()]
    keys = mx.random.normal((1, 2, rows, width))
    values = mx.random.normal((1, 2, rows, width))
    assert isinstance(caches[0], KVCache)
    caches[0].update_and_fetch(keys, values)
    recurrent = caches[1]
    assert isinstance(recurrent, DeltaCache)
    recurrent.offset = rows
    recurrent.state = mx.random.normal((1, 4, 16, 16))
    recurrent.window = mx.random.normal((1, 3, 32))
    caches[2].offset = rows
    mx.eval([tensor for layer in caches for tensor in layer.tensors])
    return caches


def test_a_round_trip_returns_the_same_tensors(tmp_path: Path) -> None:
    path = tmp_path / "entry.safetensors"
    written = filled(601)
    cache_file.dump(written, path)

    read = [KVCache(), DeltaCache(), SharedKVReader()]
    cache_file.restore(read, path)

    assert [layer.offset for layer in read] == [601, 601, 601]
    stored, back = written[0], read[0]
    assert isinstance(stored, KVCache) and isinstance(back, KVCache)
    for one, other in zip(stored.fetch(), back.fetch(), strict=True):
        assert mx.array_equal(one, other).item()
    source, target = written[1], read[1]
    assert isinstance(source, DeltaCache) and isinstance(target, DeltaCache)
    assert source.state is not None and target.state is not None
    assert mx.array_equal(source.state, target.state).item()
    assert source.window is not None and target.window is not None
    assert mx.array_equal(source.window, target.window).item()


def test_the_reserved_rows_do_not_reach_the_file(tmp_path: Path) -> None:
    """The buffer grows in blocks of 256, so a cache 601 rows deep has reserved 768. Writing
    the buffers whole would put 167 rows of zero per layer in the file — and read them back
    as state the model never wrote, at an offset that says otherwise."""
    path = tmp_path / "entry.safetensors"
    written = filled(601)
    attention = written[0]
    assert isinstance(attention, KVCache)

    cache_file.dump(written, path)

    rows = 601 * 2 * 8 * 4
    assert cache_file.written(written) == 2 * rows + (1 * 4 * 16 * 16 * 4) + (1 * 3 * 32 * 4)
    assert attention.nbytes == 2 * 768 * 2 * 8 * 4, "the buffers are the reserved ones"
    assert path.stat().st_size < attention.nbytes
    assert abs(path.stat().st_size - cache_file.written(written)) < 4096, "header aside"


def test_a_shared_reader_writes_nothing_and_comes_back_linked_by_the_trunk(
    tmp_path: Path,
) -> None:
    """It borrows the storing layer's rows, republished on every forward. Storing them again
    would put the same bytes in the file twice and restore a copy that stops tracking — so
    what it carries is its offset, and the link is the trunk's to make."""
    path = tmp_path / "entry.safetensors"
    cache_file.dump(filled(64), path)

    read = [KVCache(), DeltaCache(), SharedKVReader()]
    cache_file.restore(read, path)

    borrower = read[2]
    assert isinstance(borrower, SharedKVReader)
    assert borrower.offset == 64
    assert (borrower.keys, borrower.values) == (None, None)


def test_a_cache_that_cannot_be_written_says_so_instead_of_writing_nothing() -> None:
    """The base refuses rather than answering with an empty dict: a layer that wrote nothing
    and reported a full offset restores a trunk that decodes fluently off state it never
    had."""
    assert cache_file.storable([KVCache(), DeltaCache(), ConvCache(), SharedKVReader()])
    with pytest.raises(NotImplementedError):
        LayerCache().stored()
    with pytest.raises(NotImplementedError):
        LayerCache().restore(0, {})


def test_a_compiled_buffer_files_what_it_wrote_and_not_what_it_was_traced_with(
    tmp_path: Path,
) -> None:
    """`offset` on a fixed cache is the value the decode was traced with and stops moving;
    the rows are in `position`. Writing the first would file the prompt and drop every token
    the compiled decode produced after it."""
    growing = KVCache()
    growing.update_and_fetch(mx.random.normal((1, 2, 6, 4)), mx.random.normal((1, 2, 6, 4)))
    fixed = FixedKVCache.promote(growing, capacity=32)
    fixed.update_and_fetch(mx.random.normal((1, 2, 1, 4)), mx.random.normal((1, 2, 1, 4)))
    assert (fixed.offset, fixed.rows) == (6, 7)

    path = tmp_path / "compiled.safetensors"
    cache_file.dump([fixed], path)
    assert cache_file.read_offsets(path) == [7]

    # Read back by a growing cache, which is the point: the fixed buffer is a `KVCache` that
    # stopped growing, not another representation of one.
    read = KVCache()
    cache_file.restore([read], path)
    assert read.offset == 7
    keys, values = read.fetch()
    assert mx.allclose(keys, fixed.state[0][..., :7, :]).item()
    assert mx.allclose(values, fixed.state[1][..., :7, :]).item()


def test_a_ring_files_its_rotation_and_a_different_window_is_a_different_key() -> None:
    """A ring's slot `j` holds absolute position `j % window`, so the same bytes under
    another window are every row at the wrong place. Same shapes, same names — the digest is
    where it is caught."""
    growing = KVCache()
    growing.update_and_fetch(mx.random.normal((1, 2, 20, 4)), mx.random.normal((1, 2, 20, 4)))
    ring = RingKVCache.promote(growing, window=8)

    assert cache_file.storable([ring])
    assert cache_file.policy([ring]) == {"layers": [[("ring", 8)]]}
    assert cache_file.policy([KVCache()]) == {}, "a dense trunk keeps the key it always had"

    read = RingKVCache(8)
    read.restore(ring.rows, ring.stored())
    assert read.offset == 20 and read.state is None
    assert mx.allclose(read.fetch()[0], ring.state[0]).item()

    material = ("m", "sha", [1, 2, 3])
    assert cache_file.key(*material[:2], cache_file.policy([ring]), material[2]) != cache_file.key(
        *material[:2], cache_file.policy([RingKVCache(16)]), material[2]
    )


def test_a_file_for_another_trunk_is_refused_rather_than_half_restored(tmp_path: Path) -> None:
    """A cache of three layers read into a trunk of two would leave the two filled and the
    caller believing it had a whole cache. The list comes back untouched, which is what lets
    the caller prefill on top of it."""
    path = tmp_path / "entry.safetensors"
    cache_file.dump(filled(64), path)
    read = [KVCache(), DeltaCache()]

    with pytest.raises(cache_file.UnreadableCache):
        cache_file.restore(read, path)

    assert [layer.offset for layer in read] == [0, 0]


def test_a_truncated_file_is_refused_and_not_an_exception_the_caller_cannot_name(
    tmp_path: Path,
) -> None:
    """The net under the rename, not the defence: a file cut in half is a miss, and a miss is
    a prefill, which is always correct."""
    path = tmp_path / "entry.safetensors"
    cache_file.dump(filled(64), path)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])

    with pytest.raises(cache_file.UnreadableCache):
        cache_file.restore([KVCache(), DeltaCache(), SharedKVReader()], path)


def test_the_write_is_atomic_and_leaves_no_staging_behind(tmp_path: Path) -> None:
    """A reader must never open a half-written cache, so the bytes land under another name
    and the rename is what publishes them."""
    path = tmp_path / "entry.safetensors"

    cache_file.dump(filled(64), path)

    assert [item.name for item in tmp_path.iterdir()] == ["entry.safetensors"]


def test_the_key_moves_with_everything_that_changes_what_the_bytes_mean() -> None:
    """Four inputs, and a file read under the wrong value of any of them is the wrong cache
    arriving without an error — which a restart makes permanent."""
    base = ("m", "stamp", {"kv": None}, [1, 2, 3])
    same = cache_file.key(*base)

    assert cache_file.key("other", "stamp", {"kv": None}, [1, 2, 3]) != same
    assert cache_file.key("m", "moved", {"kv": None}, [1, 2, 3]) != same
    assert cache_file.key("m", "stamp", {"kv": "affine-4"}, [1, 2, 3]) != same
    assert cache_file.key("m", "stamp", {"kv": None}, [1, 2, 4]) != same
    assert cache_file.key(*base) == same
