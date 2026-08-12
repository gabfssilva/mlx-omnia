"""The fused counting sort against the op chain it replaces (`argsort` + take + divide).

Every output is an integer permutation product, so every assertion is exact equality —
there is no tolerance to hide behind. The reference is a *stable* `argsort` of the expert
ids, because stability is the whole contract: the sorted stream has to keep each expert's
tokens in the order they arrived, and `row_order`, `sorted_keys` and `inverse_order` are
only consistent with the gather that follows if all three come from the same permutation.

Two distributions are exercised on purpose: a uniform draw, and one where a few experts
take almost everything and most take nothing — the empty and the overloaded key are where
a base table computed by prefix sum goes wrong without the counts ever disagreeing.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia.engine.core.kernels.route.sort import (
    _SOURCE,
    _TILE,
    route_counting_sort,
    route_counting_sort_applies,
)
from mlx_omnia.engine.core.mxcompat import metal_kernel

CASES = [(64, 8, 256), (64, 4, 64), (16, 8, 256)]


def op_chain(keys: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return (
        (order // top_k).astype(np.uint32),
        keys[order].astype(np.uint32),
        inverse.astype(np.uint32),
    )


def uniform_keys(tokens: int, top_k: int, experts: int, seed: int) -> np.ndarray:
    """`top_k` distinct experts per token, which is what a real router emits."""
    rng = np.random.default_rng(seed)
    picks = np.stack([rng.choice(experts, size=top_k, replace=False) for _ in range(tokens)])
    return picks.reshape(-1).astype(np.uint32)


def skewed_keys(tokens: int, top_k: int, experts: int, seed: int) -> np.ndarray:
    """Three experts take almost every token; the rest stay empty."""
    rng = np.random.default_rng(seed)
    hot = np.array([0, experts // 2, experts - 1])
    rows: list[np.ndarray] = []
    for _ in range(tokens):
        cold = rng.choice(experts, size=top_k - 3, replace=False)
        rows.append(np.concatenate([hot, cold]))
    return np.stack(rows).reshape(-1).astype(np.uint32)


def test_applies_predicate() -> None:
    """One thread per expert id, one threadgroup per 128-slot tile."""
    assert route_counting_sort_applies(512, 256, 8)
    assert route_counting_sort_applies(256, 64, 4)
    assert not route_counting_sort_applies(500, 256, 8)
    assert not route_counting_sort_applies(0, 256, 8)
    assert not route_counting_sort_applies(512, 250, 8)
    assert not route_counting_sort_applies(512, 2048, 8)


@pytest.mark.parametrize("case", CASES)
def test_uniform_matches_the_op_chain(case: tuple[int, int, int]) -> None:
    tokens, top_k, experts = case
    keys = uniform_keys(tokens, top_k, experts, 0)
    assert keys.size % _TILE == 0
    row_order, sorted_keys, inverse = route_counting_sort(mx.array(keys), experts, top_k)
    expected = op_chain(keys, top_k)
    for ours, theirs in zip((row_order, sorted_keys, inverse), expected, strict=True):
        assert np.array(ours).tolist() == theirs.tolist()


@pytest.mark.parametrize("case", CASES)
def test_skewed_matches_the_op_chain(case: tuple[int, int, int]) -> None:
    tokens, top_k, experts = case
    keys = skewed_keys(tokens, top_k, experts, 1)
    row_order, sorted_keys, inverse = route_counting_sort(mx.array(keys), experts, top_k)
    expected = op_chain(keys, top_k)
    for ours, theirs in zip((row_order, sorted_keys, inverse), expected, strict=True):
        assert np.array(ours).tolist() == theirs.tolist()


def test_inverse_order_undoes_the_gather() -> None:
    """The three outputs are one permutation seen three ways: unsorting the sorted keys
    with `inverse_order` has to give the input stream back."""
    tokens, top_k, experts = 64, 8, 256
    keys = uniform_keys(tokens, top_k, experts, 2)
    _, sorted_keys, inverse = route_counting_sort(mx.array(keys), experts, top_k)
    assert np.array(sorted_keys)[np.array(inverse)].tolist() == keys.tolist()


_MUTATIONS = {
    "walk the tile backwards": (
        "    uint idx = t * TILE + i;",
        "    uint idx = t * TILE + (TILE - 1 - i);",
    ),
    "forget the keys in earlier tiles": (
        "uint off = simd_base + lane_excl +\n"
        "    atomic_load_explicit(&tg_before[k], memory_order_relaxed);",
        "uint off = simd_base + lane_excl;",
    ),
    "drop the fan-out from the gathered row": (
        "        row_order[off] = idx / M;",
        "        row_order[off] = idx;",
    ),
}


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_mutations_break_the_permutation(mutation: str) -> None:
    """Stability is what the backwards walk breaks, and it only shows when an expert is
    picked more than once inside one tile — the skewed fixture guarantees that."""
    old, new = _MUTATIONS[mutation]
    source = _SOURCE.replace(old, new)
    assert source != _SOURCE
    tokens, top_k, experts = 64, 8, 256
    keys = skewed_keys(tokens, top_k, experts, 3)
    expected = op_chain(keys, top_k)
    broken = metal_kernel(
        name=f"route_counting_sort_broken_{sorted(_MUTATIONS).index(mutation)}",
        input_names=["keys"],
        output_names=["row_order", "sorted_keys", "inverse_order"],
        source=source,
    )(
        inputs=[mx.array(keys)],
        template=[("TILE_SIZE", _TILE), ("TOPK", top_k), ("NKEYS", experts)],
        grid=((keys.size // _TILE) * experts, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(keys.size,), (keys.size,), (keys.size,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
    )
    assert any(
        np.array(ours).tolist() != theirs.tolist()
        for ours, theirs in zip(broken, expected, strict=True)
    )


def test_unmutated_replica_of_the_dispatch_agrees_with_the_module() -> None:
    tokens, top_k, experts = 64, 8, 256
    keys = skewed_keys(tokens, top_k, experts, 3)
    expected = route_counting_sort(mx.array(keys), experts, top_k)
    intact = metal_kernel(
        name="route_counting_sort_intact",
        input_names=["keys"],
        output_names=["row_order", "sorted_keys", "inverse_order"],
        source=_SOURCE,
    )(
        inputs=[mx.array(keys)],
        template=[("TILE_SIZE", _TILE), ("TOPK", top_k), ("NKEYS", experts)],
        grid=((keys.size // _TILE) * experts, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(keys.size,), (keys.size,), (keys.size,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
    )
    for ours, theirs in zip(intact, expected, strict=True):
        assert mx.array_equal(ours, theirs)
