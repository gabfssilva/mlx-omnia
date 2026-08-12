import mlx.core as mx

from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, RingKVCache


def _cache(length: int) -> KVCache:
    cache = KVCache()
    rows = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(rows, rows + 10)
    mx.eval(cache.tensors)
    return cache


def test_fixed_cache_promotion_preserves_rows_and_offset() -> None:
    cache = FixedKVCache.promote(_cache(5), capacity=8)

    keys, values = cache.fetch()
    assert keys.shape == (1, 1, 8, 1)
    assert keys[..., :5, :].tolist() == [[[[0.0], [1.0], [2.0], [3.0], [4.0]]]]
    assert values[..., :5, :].tolist() == [[[[10.0], [11.0], [12.0], [13.0], [14.0]]]]
    assert cache.position.item() == 5


def test_ring_promotion_places_absolute_rows_at_their_ring_indices() -> None:
    cache = RingKVCache.promote(_cache(6), window=4)

    keys, values = cache.fetch()
    assert keys.reshape(-1).tolist() == [4.0, 5.0, 2.0, 3.0]
    assert values.reshape(-1).tolist() == [14.0, 15.0, 12.0, 13.0]
    assert cache.position.item() == 6
    assert cache.write_index.item() == 2


def test_fixed_cache_state_advances_inside_compiled_function() -> None:
    cache = FixedKVCache.promote(_cache(2), capacity=5)

    def append(row: mx.array) -> mx.array:
        keys, _ = cache.update_and_fetch(row, row)
        return keys

    compiled = mx.compile(append, inputs=cache.state, outputs=cache.state)
    for value in (2.0, 3.0):
        keys = compiled(mx.array([[[[value]]]], dtype=mx.float32))
        mx.eval(keys)

    assert keys.reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]
    assert cache.position.item() == 4
