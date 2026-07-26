"""softmax -> top-k -> renormalize in one dispatch, against the op chain (`moe_route`).

The replica is the routing step the kernel replaces — `softmax(logits, precise=True)`,
top-k, renormalize — over random logits, plus the shared-expert slot qwen3_5 appends
(one extra logit, its sigmoid riding as slot k). No checkpoint: the kernel only sees a
router row, and a random one exercises it as well as a real one.

What is exact and what is not, kept apart on purpose:

* the *selection* is bit-exact and is asserted with `mx.array_equal`. Its promised
  semantics is a stable descending order — ties go to the higher index, matching
  mlx's stable ascending `argpartition` — so the reference is a stable `argsort`
  reversed, never `argpartition` (whose order among equals is not the contract).
  The tie cases below are built from *exactly equal* logits, so both sides see
  bitwise equal probabilities and the tie rule is the only thing under test; ties
  are placed both inside one lane (indices 32 apart) and across lanes, and the top-k
  cut falls inside the tie group.
* the *weights* are not bit-exact and are not asserted to be: the kernel rounds each
  probability and each partial of the renormalizing sum in T, in its own order, and
  its `exp` is Metal's. The bound is the dtype's own rounding — a handful of T ulps
  on the largest weight, which is what `relative_diff` normalizes by — and the
  comparison side runs in fp32.

For the random logits the top-k is bumped clear of the rest, so the cut is never a
near-tie the two implementations could split; equalities among the picked ones are
harmless because the picked set is compared sorted.
"""

import mlx.core as mx
import numpy as np
import pytest
from conftest import relative_diff

from sideros.core.kernels.moe_route import _SOURCE, softmax_topk, softmax_topk_applies
from sideros.core.mxcompat import metal_kernel, softmax

SHAPES = [(32, 4), (128, 8), (256, 8)]
SHARED_LOGIT = 1.5


def _ulps(dtype: mx.Dtype, count: int) -> float:
    return count * (2.0**-23 if dtype == mx.float32 else 2.0**-8)


def _weight_bound(dtype: mx.Dtype) -> float:
    """A handful of T ulps, plus the fp32 slack the two `exp`s cost: the kernel rounds
    every probability and every partial of the renormalizing sum in T, in the opposite
    order from the op chain, and its exponential is Metal's, not mlx's."""
    return _ulps(dtype, 8) + 32 * 2.0**-23


def replica(experts: int, k: int, seed: int, dtype: mx.Dtype, *, shared: bool) -> mx.array:
    """Random router logits whose top-k stands clear of the rest by two whole logits."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(experts)
    logits[np.argsort(logits, kind="stable")[-k:]] += 2.0
    if shared:
        logits = np.concatenate([logits, [SHARED_LOGIT]])
    return mx.array(logits.astype(np.float32)).astype(dtype)


def op_chain(logits: mx.array, k: int, *, shared: bool) -> tuple[mx.array, mx.array]:
    """The picked indices in the kernel's promised order and the renormalized weights."""
    experts = logits.size - (1 if shared else 0)
    probs = np.array(softmax(logits[:experts], axis=-1, precise=True).astype(mx.float32))
    chosen = np.argsort(probs, kind="stable")[::-1][:k]
    weights = probs[chosen] / probs[chosen].sum()
    if shared:
        chosen = np.concatenate([chosen, [experts]])
        weights = np.concatenate([weights, [_shared_weight(logits, experts)]])
    return mx.array(chosen.astype(np.uint32)), mx.array(weights.astype(np.float32))


def _shared_weight(logits: mx.array, experts: int) -> float:
    """The kernel's own stable form of the sigmoid, evaluated in fp32."""
    x = float(logits[experts].item())
    y = 1.0 / (1.0 + float(np.exp(abs(x))))
    return y if x < 0 else 1.0 - y


def _by_index(indices: mx.array, weights: mx.array) -> dict[int, float]:
    picked = np.array(indices).tolist()
    values = np.array(weights.astype(mx.float32)).tolist()
    return dict(zip(picked, values, strict=True))


def test_applies_predicate() -> None:
    """One simdgroup owns the row and writes the k winners from its own lanes."""
    for experts, k in SHAPES:
        assert softmax_topk_applies(experts, k)
    assert not softmax_topk_applies(100, 8)
    assert not softmax_topk_applies(16, 8)
    assert not softmax_topk_applies(128, 33)
    assert not softmax_topk_applies(128, 0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_op_chain(shape: tuple[int, int], shared: bool, dtype: mx.Dtype) -> None:
    experts, k = shape
    for seed in range(4):
        logits = replica(experts, k, seed, dtype, shared=shared)
        indices, weights = softmax_topk(logits, k, shared=shared)
        ref_indices, ref_weights = op_chain(logits, k, shared=shared)
        assert indices.shape == (k + (1 if shared else 0),)
        assert mx.array_equal(mx.sort(indices), mx.sort(ref_indices))

        ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
        order = sorted(theirs)
        assert relative_diff(
            mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
        ) < _weight_bound(dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_weights_come_out_descending_and_summing_to_one(
    shape: tuple[int, int], dtype: mx.Dtype
) -> None:
    """The routed slots are written in the order they were picked, renormalized over
    themselves; the shared slot is a sigmoid and belongs to neither property."""
    experts, k = shape
    _, weights = softmax_topk(replica(experts, k, 5, dtype, shared=True), k, shared=True)
    routed = np.array(weights[:k].astype(mx.float32))
    assert (np.diff(routed) <= 0.0).all()
    assert abs(float(routed.sum()) - 1.0) < _weight_bound(dtype)


def tied_logits(dtype: mx.Dtype, *, shared: bool) -> tuple[mx.array, int]:
    """128 strictly decreasing logits with one group of six forced exactly equal.

    The group sits at rank 4 and spans six ranks, so the k=8 cut falls inside it: five
    of the six are kept and which one is dropped is the tie rule's alone to decide (the
    lowest index, 3, and every pick's position in the output too). Indices 3/35/67/99
    are the same lane
    (four slots of one thread), 11 and 120 are two other lanes — the rule has to hold
    both inside a lane's registers and across the simd_max.
    """
    values = np.arange(128, dtype=np.float64) * -0.5
    values[[3, 11, 35, 67, 99, 120]] = values[3]
    if shared:
        values = np.concatenate([values, [SHARED_LOGIT]])
    return mx.array(values.astype(np.float32)).astype(dtype), 8


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
def test_exact_ties_go_to_the_higher_index(dtype: mx.Dtype, shared: bool) -> None:
    logits, k = tied_logits(dtype, shared=shared)
    indices, _ = softmax_topk(logits, k, shared=shared)
    ref_indices, _ = op_chain(logits, k, shared=shared)
    expected = [0, 1, 2, 120, 99, 67, 35, 11] + ([128] if shared else [])
    assert np.array(ref_indices).tolist() == expected
    assert mx.array_equal(indices, ref_indices)


_MUTATIONS = {
    "resolve a lane's tie downwards": (
        "for (int i = (int)per_lane - 1; i >= 0; i--) {",
        "for (int i = 0; i < (int)per_lane; i++) {",
    ),
    "keep picking the same expert": (
        "if (winner == cand && (int)i == slot) p[i] = -1.0f;",
        "if (false) p[i] = -1.0f;",
    ),
    "drop the renormalization": (
        "if (lane < TOPK) OW[lane] = (T)(pick[lane] / total);",
        "if (lane < TOPK) OW[lane] = (T)pick[lane];",
    ),
    "drop the shared sigmoid's negative branch": (
        "OW[TOPK] = (sx < 0) ? sy : 1 - sy;",
        "OW[TOPK] = sy;",
    ),
}


def _route(source: str, name: str, logits: mx.array, k: int, *, shared: bool) -> list[mx.array]:
    experts = logits.size - (1 if shared else 0)
    slots = k + (1 if shared else 0)
    return metal_kernel(
        name=name, input_names=["L"], output_names=["OI", "OW"], source=source
    )(
        inputs=[logits],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", experts), ("SHARED", int(shared))],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(slots,), (slots,)],
        output_dtypes=[mx.uint32, logits.dtype],
    )


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_mutations_break_the_routing(mutation: str) -> None:
    """Each documented break is compiled as its own kernel; the tie case is the one
    that exercises all four, so both selection and weights are checked against it."""
    old, new = _MUTATIONS[mutation]
    logits, k = tied_logits(mx.float32, shared=True)
    source = _SOURCE.replace(old, new)
    assert source != _SOURCE
    ref_indices, ref_weights = op_chain(logits, k, shared=True)
    broken = _route(source, f"moe_route_broken_{sorted(_MUTATIONS).index(mutation)}",
                    logits, k, shared=True)
    picked_changed = not mx.array_equal(broken[0], ref_indices)
    weights_changed = relative_diff(broken[1], ref_weights) > 1e-2
    assert picked_changed or weights_changed


def test_unmutated_replica_of_the_dispatch_agrees_with_the_module() -> None:
    """The dispatch replica above is only evidence if the untouched source reproduces
    what the module's own entry point returns."""
    logits, k = tied_logits(mx.float32, shared=True)
    indices, weights = softmax_topk(logits, k, shared=True)
    intact = _route(_SOURCE, "moe_route_intact", logits, k, shared=True)
    assert mx.array_equal(intact[0], indices)
    assert mx.array_equal(intact[1], weights)
