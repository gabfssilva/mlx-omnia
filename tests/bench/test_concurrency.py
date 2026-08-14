from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.bench.arms.omnia import measure_concurrency
from mlx_omnia.bench.gate import Cool
from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache


class CountingModel:
    continuous_batching = True

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[KVStore]) -> mx.array:
        self.batch_sizes.append(ids.shape[0])
        targets = (ids + 1) % 32
        return -mx.abs(mx.arange(32) - targets[..., None]).astype(mx.float32)


def test_sweep_measures_requested_concurrencies_through_batched_steps() -> None:
    model = CountingModel()

    result = measure_concurrency(
        model,
        [1, 2],
        concurrencies=[1, 2, 4, 8],
        tokens=3,
        runs=2,
        gate=Cool(None),
    )

    assert [row.concurrency for row in result.rows] == [1, 2, 4, 8]
    assert all(row.aggregate_tps > 0 for row in result.rows)
    assert all(row.per_request_tps == row.aggregate_tps / row.concurrency for row in result.rows)
    assert {1, 2, 4, 8}.issubset(model.batch_sizes)
    assert result.rows[0].speedup == 1.0
    assert result.rows[0].efficiency == 1.0
    assert "vs CB C1" in result.render()
    assert all(len(row.samples) == 2 for row in result.rows)
    assert all(len(row.kv_bytes) == 2 for row in result.rows)
    assert "min" in result.render() and "max" in result.render() and "n=2" in result.render()
    assert "KV" in result.render()
