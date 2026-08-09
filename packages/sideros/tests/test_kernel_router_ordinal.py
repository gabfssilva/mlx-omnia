"""The bitonic tournament against the op chain (`sigmoid` -> bias -> stable sort).

Selection is asserted for exact index equality, not for value closeness — the tie rule is
what breaks a sorting network, and it is the opposite of `moe_route`'s: an equal ordinal
resolves to the *lower* index, which makes the pick a stable descending sort of
`score + bias`. The reference is therefore a stable `argsort` of the negated biased
scores, never `argpartition`, whose order among equals is not a contract.

The tie fixture forces six exactly equal logits with the top-k cut falling inside the
group, so five are kept and which one is dropped is the tie rule's alone to decide. The
group is spread over four of the eight 32-lane blocks, so it exercises the rule inside
one block's own bitonic run, in the per-block extraction, and in the 64-candidate merge
that crosses the two simdgroups.

The random rows are spaced clear of any near-tie: the sigmoid is monotone, so spacing the
logits spaces the scores and neither side can split a cut the other keeps. The weights
are the winners' *unbiased* scores renormalized over themselves, compared with the house
metric rather than bitwise — the kernel's exponential is Metal's.
"""

import mlx.core as mx
import numpy as np
import pytest
from conftest import relative_diff

from sideros.core.kernels.router_ordinal import (
    _SOURCE,
    ORDINAL_HEADER,
    router_tournament,
    router_tournament_applies,
)
from sideros.core.mxcompat import metal_kernel

EXPERTS = 256
TOPK = 8
WEIGHT_BOUND = 1e-5


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    y = 1.0 / (1.0 + np.exp(np.abs(x)))
    return np.where(x < 0.0, y, 1.0 - y)


def op_chain(logits: mx.array, bias: mx.array, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable descending order of `sigmoid(logit) + bias` — ties to the lower index — and
    the unbiased scores of the winners, renormalized."""
    rows = np.array(logits.astype(mx.float32), dtype=np.float64).reshape(-1, EXPERTS)
    scores = _stable_sigmoid(rows)
    biased = scores + np.array(bias, dtype=np.float64)
    chosen = np.argsort(-biased, kind="stable", axis=-1)[:, :k]
    picked = np.take_along_axis(scores, chosen, axis=-1)
    return chosen.astype(np.uint32), picked / picked.sum(axis=-1, keepdims=True)


def separated_logits(rows: int, seed: int, dtype: mx.Dtype) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal((rows, EXPERTS)) * 2.0).astype(np.float32)).astype(dtype)


def graded_bias(logits: mx.array, seed: int) -> mx.array:
    """A bias that puts `score + bias` on a randomly permuted grid four apart.

    The kernel scores in fp32 with Metal's `exp` while the reference scores the same
    logits in fp64. On an unconstrained random row the eighth and ninth biased scores can
    land closer than that difference, and the test would be measuring the exponential
    instead of the sorting network. A grid step of four keeps every margin the pick
    depends on clear of it — even across rows, where the scores the bias was built from
    move by at most one — while leaving the score decisive for any real error: a wrong
    sigmoid moves a value by O(1) and the reference reorders with it.
    """
    rng = np.random.default_rng(seed)
    rows = np.array(logits.astype(mx.float32), dtype=np.float64).reshape(-1, EXPERTS)
    scores = _stable_sigmoid(rows[0])
    grid = np.empty(EXPERTS)
    grid[rng.permutation(EXPERTS)] = (np.arange(EXPERTS) - EXPERTS / 2) * 4.0
    return mx.array((grid - scores).astype(np.float32))


def small_bias(seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed).standard_normal(EXPERTS).astype(np.float32) * 0.1)


def _boost(expert: int) -> np.ndarray:
    """Enough to make one expert the outright winner without disturbing the grid order."""
    return np.where(np.arange(EXPERTS) == expert, 1e4, 0.0).astype(np.float32)


def tied_logits(dtype: mx.Dtype) -> mx.array:
    """256 strictly decreasing logits with one group of six forced exactly equal.

    The group sits at rank 3 and spans six ranks, so the k=8 cut falls inside it: five of
    the six are kept and the sixth (index 120, the highest) is the one the rule drops.
    Indices 3 and 11 share the first 32-lane block, 35 and 67 land in two more, and 99
    and 120 share a fourth — so the rule is exercised within a block, in the per-block
    extraction, and across the merge that joins the two simdgroups.
    """
    values = np.arange(EXPERTS, dtype=np.float64) * -0.5
    values[[3, 11, 35, 67, 99, 120]] = values[3]
    return mx.array(values.astype(np.float32)).astype(dtype).reshape(1, EXPERTS)


def test_applies_predicate() -> None:
    """The 64-candidate merge is written out for exactly two simdgroups."""
    assert router_tournament_applies(256, 8)
    assert router_tournament_applies(128, 16)
    assert router_tournament_applies(512, 4)
    assert not router_tournament_applies(256, 6)
    assert not router_tournament_applies(64, 8)
    assert not router_tournament_applies(250, 8)
    assert not router_tournament_applies(2048, 1)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 5])
def test_matches_op_chain(rows: int, dtype: mx.Dtype) -> None:
    for seed in range(4):
        logits = separated_logits(rows, seed, dtype)
        bias = graded_bias(logits, seed + 100)
        indices, scores = router_tournament(logits, bias, TOPK)
        ref_indices, ref_scores = op_chain(logits, bias, TOPK)
        assert indices.shape == (rows, TOPK) == scores.shape
        assert np.array(indices).tolist() == ref_indices.tolist()
        assert relative_diff(scores, mx.array(ref_scores.astype(np.float32))) < WEIGHT_BOUND


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_weights_are_descending_and_sum_to_one(dtype: mx.Dtype) -> None:
    logits = separated_logits(1, 7, dtype)
    _, scores = router_tournament(logits, mx.zeros((EXPERTS,)), TOPK)
    row = np.array(scores[0].astype(mx.float32))
    assert (np.diff(row) <= 0.0).all()
    assert abs(float(row.sum()) - 1.0) < WEIGHT_BOUND


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_exact_ties_go_to_the_lower_index(dtype: mx.Dtype) -> None:
    logits = tied_logits(dtype)
    zero = mx.zeros((EXPERTS,))
    indices, _ = router_tournament(logits, zero, TOPK)
    ref_indices, _ = op_chain(logits, zero, TOPK)
    assert ref_indices[0].tolist() == [0, 1, 2, 3, 11, 35, 67, 99]
    assert np.array(indices).tolist() == ref_indices.tolist()


def test_bias_selects_but_never_weighs() -> None:
    """Boosting one outsider's bias pulls it in, while its weight still comes from the
    unbiased score — the biased value must not leak into the renormalization."""
    logits = separated_logits(1, 3, mx.bfloat16)
    bias = graded_bias(logits, 300)
    base, _ = router_tournament(logits, bias, TOPK)
    outsider = next(e for e in range(EXPERTS) if e not in set(np.array(base[0]).tolist()))
    boosted = mx.array(np.array(bias) + _boost(outsider))
    indices, scores = router_tournament(logits, boosted, TOPK)
    ref_indices, ref_scores = op_chain(logits, boosted, TOPK)
    assert int(np.array(indices)[0][0]) == outsider
    assert np.array(indices).tolist() == ref_indices.tolist()
    assert relative_diff(scores, mx.array(ref_scores.astype(np.float32))) < WEIGHT_BOUND


_MUTATIONS = {
    "resolve a tie to the higher index": (
        "    return a_index < b_index;",
        "    return a_index > b_index;",
    ),
    "keep the wrong end of a descending block": (
        "bool is_local_top8 = block_ascending ? (within_block < top_k)\n"
        "                                     : (within_block >= 32 - top_k);",
        "bool is_local_top8 = (within_block < top_k);",
    ),
    "skip the merge that crosses the simdgroups": (
        "    uint partner = lane ^ 32u;",
        "    uint partner = lane;",
    ),
    "ignore the correction bias": (
        "float key = -(score + float(correction_bias[lane]));",
        "float key = -score;",
    ),
    "drop the renormalization": (
        "    router_scores[row * top_k + lane] = my_score2 / total;",
        "    router_scores[row * top_k + lane] = my_score2;",
    ),
}


def _run(source: str, header: str, name: str, logits: mx.array, bias: mx.array) -> list[mx.array]:
    rows = logits.size // EXPERTS
    return metal_kernel(
        name=name,
        input_names=["logits", "correction_bias"],
        output_names=["router_indices", "router_scores"],
        source=source,
        header=header,
    )(
        inputs=[logits, bias],
        template=[("EXPERTS", EXPERTS), ("TOPK", TOPK)],
        grid=(EXPERTS, rows, 1),
        threadgroup=(EXPERTS, 1, 1),
        output_shapes=[(rows, TOPK), (rows, TOPK)],
        output_dtypes=[mx.uint32, mx.float32],
    )


def _mutation_cases() -> list[tuple[mx.array, mx.array]]:
    """Two fixtures: the tie group with no bias, which drives the ordering rules, and a
    separated row whose bias decides the pick, which drives the two that only show up
    when the key is not the score."""
    graded = separated_logits(1, 3, mx.float32)
    return [
        (tied_logits(mx.float32), mx.zeros((EXPERTS,))),
        (graded, mx.array(np.array(graded_bias(graded, 300)) + _boost(200))),
    ]


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_mutations_break_the_routing(mutation: str) -> None:
    old, new = _MUTATIONS[mutation]
    source, header = _SOURCE.replace(old, new), ORDINAL_HEADER.replace(old, new)
    assert (source, header) != (_SOURCE, ORDINAL_HEADER)
    index = sorted(_MUTATIONS).index(mutation)
    broke = False
    for case, (logits, bias) in enumerate(_mutation_cases()):
        ref_indices, ref_scores = op_chain(logits, bias, TOPK)
        got = _run(source, header, f"router_tournament_broken_{index}_{case}", logits, bias)
        broke = broke or np.array(got[0]).tolist() != ref_indices.tolist()
        broke = broke or (
            relative_diff(got[1], mx.array(ref_scores.astype(np.float32))) > 1e-2
        )
    assert broke


def test_unmutated_replica_of_the_dispatch_agrees_with_the_module() -> None:
    """The mutation dispatches are only evidence if the untouched source, launched the
    same way by hand, reproduces what the module returns."""
    logits = tied_logits(mx.float32)
    bias = small_bias(11)
    indices, scores = router_tournament(logits, bias, TOPK)
    intact = _run(_SOURCE, ORDINAL_HEADER, "router_tournament_intact", logits, bias)
    assert mx.array_equal(intact[0], indices)
    assert mx.array_equal(intact[1], scores)
