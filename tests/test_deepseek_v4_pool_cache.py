"""PoolCache's window bookkeeping: what it hands the compressor must be the same stream
the compressor would have seen from a single call, whatever the calls are split into."""

import mlx.core as mx
import pytest

from mlx_omnia.engine.models.deepseek_v4.layers.cache import DeepseekV4Cache, PoolCache

RATIO = 4
DIM = 3


def _stream(length: int) -> tuple[mx.array, mx.array]:
    total = mx.arange(length * DIM, dtype=mx.float32).reshape(1, length, DIM)
    return total, total + 1000.0


def _feed(cache: PoolCache, splits: list[int]) -> tuple[list[mx.array], list[int]]:
    kv, gate = _stream(sum(splits))
    ready: list[mx.array] = []
    bases: list[int] = []
    offset = 0
    for length in splits:
        chunk = slice(offset, offset + length)
        ready_kv, ready_gate, base = cache.accumulate(kv[:, chunk], gate[:, chunk], offset)
        if ready_kv.shape[1]:
            assert mx.array_equal(ready_gate, ready_kv + 1000.0)
            ready.append(ready_kv)
            bases.append(base)
        offset += length
    return ready, bases


@pytest.mark.parametrize(
    "splits",
    [
        [1] * 10,
        [7, 1, 1, 1],
        [4, 1, 1, 1, 1],
        [2, 3, 1, 1, 5],
        [9],
        [1, 8, 1],
    ],
)
def test_windows_match_the_whole_stream(splits: list[int]) -> None:
    total = sum(splits)
    ready, bases = _feed(PoolCache(RATIO), splits)
    whole, _ = _stream(total)

    assert mx.array_equal(mx.concatenate(ready, axis=1), whole[:, : total // RATIO * RATIO])

    # Each block is announced at the absolute position of its first row: the blocks tile the
    # stream, so that is the number of rows handed over before it.
    expected, written = [], 0
    for rows in ready:
        assert rows.shape[1] % RATIO == 0
        expected.append(written)
        written += rows.shape[1]
    assert bases == expected


def test_remainder_is_the_tail_of_the_stream() -> None:
    cache = PoolCache(RATIO)
    _feed(cache, [3, 1, 1, 1])
    whole, gate = _stream(6)

    assert cache.remainder == 6 % RATIO
    assert cache.tail_kv is not None and cache.tail_gate is not None
    assert mx.array_equal(cache.tail_kv[:, : cache.remainder], whole[:, -(6 % RATIO) :])
    assert mx.array_equal(cache.tail_gate[:, : cache.remainder], gate[:, -(6 % RATIO) :])


def test_pooled_rows_grow_and_the_checkpoint_rewinds_them() -> None:
    cache = PoolCache(RATIO)
    row = mx.ones((1, 1, 1, DIM))
    cache.append(row)
    restore = cache.checkpoint()
    cache.append(row * 2)

    assert cache.pooled_rows == 2
    assert mx.array_equal(cache.fetch(DIM, mx.float32), mx.concatenate([row, row * 2], axis=2))

    restore()
    assert cache.pooled_rows == 1
    assert mx.array_equal(cache.fetch(DIM, mx.float32), row)


def test_a_restored_pool_is_not_the_pool_the_replay_would_need() -> None:
    """The rewind speculation takes when a cache cannot trim is `checkpoint()` plus a
    replay. A pool cannot serve it: the round overwrote the tail rows the replay reads, so
    the layer has to refuse rather than pool a window out of tokens that never happened."""
    kv, gate = _stream(12)

    def chunk(cache: PoolCache, start: int, stop: int) -> mx.array:
        ready, _, _ = cache.accumulate(kv[:, start:stop], gate[:, start:stop], start)
        return ready

    speculative = PoolCache(RATIO)
    chunk(speculative, 0, 2)
    restore = speculative.checkpoint()
    chunk(speculative, 2, 7)  # the round: a window completes and three rows land in the tail
    restore()
    replayed = chunk(speculative, 2, 5)  # the rewind: only the accepted rows run again

    honest = PoolCache(RATIO)
    chunk(honest, 0, 2)
    assert not mx.array_equal(replayed, chunk(honest, 2, 5))

    assert speculative.is_replayable is False
    assert DeepseekV4Cache(RATIO, indexed=True).is_replayable is False
    assert DeepseekV4Cache(0, indexed=False).is_replayable is True


def test_mask_hides_rows_a_position_cannot_see_yet() -> None:
    cache = PoolCache(RATIO)
    cache.append(mx.ones((1, 1, 3, DIM)))

    assert cache.mask(1, 7) is None
    visible = cache.mask(2, 7)
    assert visible is not None
    # Positions 7 and 8: (7 + 1) // 4 == 2 rows, (8 + 1) // 4 == 2 rows.
    assert visible.tolist() == [[True, True, False], [True, True, False]]
