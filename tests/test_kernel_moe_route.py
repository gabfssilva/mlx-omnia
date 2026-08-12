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
import mlx.nn as nn
import numpy as np
import pytest
from conftest import relative_diff

from mlx_omnia.engine.core.kernels.route import (
    BiasTopkRoute,
    DefaultRoute,
    OrdinalRoute,
    Route,
    Routing,
    SigmoidTopkRoute,
    SigmoidWideRoute,
    SoftmaxTopkRoute,
)
from mlx_omnia.engine.core.kernels.route.softmax_topk import (
    _SIGMOID_SOURCE,
    _SOFTPLUS_GATHER_SOURCE,
    _SOFTPLUS_HEADER,
    _SOFTPLUS_SOURCE,
    _SOURCE,
    sigmoid_topk,
    softmax_topk,
    softmax_topk_applies,
    softplus_gather,
    softplus_topk,
)
from mlx_omnia.engine.core.mxcompat import metal_kernel, softmax

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


# --- sigmoid routing (laguna): selection by sigmoid+bias, weights unbiased ---

SIGMOID_SCALE = 2.5


def sigmoid_bias(experts: int, seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed + 100).standard_normal(experts) * 0.1).astype(
        mx.float32
    )


def sigmoid_op_chain(
    logits: mx.array, bias: mx.array, k: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """Selection on the biased fp32 scores (ties to the higher index, stable argsort
    reversed), weights from the unbiased scores: renormalized in fp32, rounded to T,
    then scaled in T — the stock chain's astype placement."""
    probs = np.array(mx.sigmoid(logits.astype(mx.float32)))
    biased = probs + np.array(bias)
    chosen = np.argsort(biased, kind="stable")[::-1][:k]
    weights = probs[chosen] / probs[chosen].sum()
    scaled = mx.array(weights.astype(np.float32)).astype(dtype) * SIGMOID_SCALE
    return mx.array(chosen.astype(np.uint32)), scaled


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_sigmoid_matches_op_chain(shape: tuple[int, int], dtype: mx.Dtype) -> None:
    experts, k = shape
    for seed in range(4):
        logits = replica(experts, k, seed, dtype, shared=False)
        bias = sigmoid_bias(experts, seed)
        indices, weights = sigmoid_topk(logits, bias, k, scale=SIGMOID_SCALE)
        ref_indices, ref_weights = sigmoid_op_chain(logits, bias, k, dtype)
        assert mx.array_equal(mx.sort(indices), mx.sort(ref_indices))
        ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
        order = sorted(theirs)
        assert relative_diff(
            mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
        ) < _weight_bound(dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sigmoid_exact_ties_go_to_the_higher_index(dtype: mx.Dtype) -> None:
    """Zero bias keeps the sigmoid monotonic in the logits, so the tie group and the
    promised order are the softmax case's."""
    logits, k = tied_logits(dtype, shared=False)
    indices, _ = sigmoid_topk(logits, mx.zeros((logits.size,)), k, scale=SIGMOID_SCALE)
    assert np.array(indices).tolist() == [0, 1, 2, 120, 99, 67, 35, 11]


def test_sigmoid_bias_selects_but_never_weighs() -> None:
    """Boosting one outsider's bias must pull it in, while its weight still comes from
    the unbiased score — the biased value must not leak into the renormalization."""
    logits = replica(256, 8, 2, mx.bfloat16, shared=False)
    zero = mx.zeros((256,))
    base_idx, _ = sigmoid_topk(logits, zero, 8, scale=SIGMOID_SCALE)
    outsider = next(e for e in range(256) if e not in set(np.array(base_idx).tolist()))
    boosted = mx.where(mx.arange(256) == outsider, 10.0, 0.0)
    indices, weights = sigmoid_topk(logits, boosted, 8, scale=SIGMOID_SCALE)
    assert outsider in set(np.array(indices).tolist())
    ref_indices, ref_weights = sigmoid_op_chain(logits, boosted, 8, mx.bfloat16)
    ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
    order = sorted(theirs)
    assert relative_diff(
        mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
    ) < _weight_bound(mx.bfloat16)


_SIGMOID_MUTATIONS = {
    "resolve a lane's tie downwards": (
        "for (int i = (int)per_lane - 1; i >= 0; i--) {",
        "for (int i = 0; i < (int)per_lane; i++) {",
    ),
    "keep picking the same expert": (
        "if (winner == cand && slot >= 0) b[(uint)slot] = -INFINITY;",
        "if (false) b[(uint)slot] = -INFINITY;",
    ),
    "drop the renormalization": (
        "T w = (T)(pick[lane] / total);",
        "T w = (T)pick[lane];",
    ),
    "ignore the bias in selection": (
        "b[i] = s[i] + B[lane + i * 32];",
        "b[i] = s[i];",
    ),
}


def _sigmoid_route(
    source: str, name: str, logits: mx.array, bias: mx.array, k: int
) -> list[mx.array]:
    return metal_kernel(
        name=name, input_names=["L", "B", "SC"], output_names=["OI", "OW"], source=source
    )(
        inputs=[logits, bias, mx.array(SIGMOID_SCALE, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", logits.size)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, logits.dtype],
    )


@pytest.mark.parametrize("mutation", sorted(_SIGMOID_MUTATIONS))
def test_sigmoid_mutations_break_the_routing(mutation: str) -> None:
    """The tie case exercises the pick loop; the boosted-outsider bias makes the bias
    term decisive, so dropping it changes the selection."""
    old, new = _SIGMOID_MUTATIONS[mutation]
    source = _SIGMOID_SOURCE.replace(old, new)
    assert source != _SIGMOID_SOURCE
    logits, k = tied_logits(mx.float32, shared=False)
    bias = mx.where(mx.arange(logits.size) == logits.size - 1, 10.0, 0.0)
    ref_indices, ref_weights = sigmoid_op_chain(logits, bias, k, mx.float32)
    broken = _sigmoid_route(
        source,
        f"moe_sigmoid_broken_{sorted(_SIGMOID_MUTATIONS).index(mutation)}",
        logits,
        bias,
        k,
    )
    picked_changed = not mx.array_equal(mx.sort(broken[0]), mx.sort(ref_indices))
    ours, theirs = _by_index(broken[0], broken[1]), _by_index(ref_indices, ref_weights)
    weights_changed = any(
        e not in theirs or abs(w - theirs[e]) > 1e-2 * abs(theirs[e]) for e, w in ours.items()
    )
    assert picked_changed or weights_changed


def test_sigmoid_unmutated_replica_agrees_with_the_module() -> None:
    logits, k = tied_logits(mx.float32, shared=False)
    bias = sigmoid_bias(logits.size, 0)
    indices, weights = sigmoid_topk(logits, bias, k, scale=SIGMOID_SCALE)
    intact = _sigmoid_route(_SIGMOID_SOURCE, "moe_sigmoid_intact", logits, bias, k)
    assert mx.array_equal(intact[0], indices)
    assert mx.array_equal(intact[1], weights)


# --- sqrt(softplus) routing (deepseek_v4): selection by score+bias, weights unbiased ---

SOFTPLUS_SCALE = 1.5


def softplus_op_chain(
    logits: mx.array, bias: mx.array, k: int, *, norm: bool
) -> tuple[mx.array, mx.array]:
    """The stock `MoEGate` chain: fp32 `sqrt(softplus)` scores, selection on the biased
    scores (ties to the higher index, stable argsort reversed), weights unbiased,
    renormalized when `norm`, scaled in fp32."""
    scores = np.array(mx.sqrt(nn.softplus(logits.astype(mx.float32))))
    biased = scores + np.array(bias)
    chosen = np.argsort(biased, kind="stable")[::-1][:k]
    weights = scores[chosen]
    if norm:
        weights = weights / (weights.sum() + 1e-20)
    scaled = (weights * SOFTPLUS_SCALE).astype(np.float32)
    return mx.array(chosen.astype(np.uint32)), mx.array(scaled)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", [(256, 6), (128, 8)])
def test_softplus_matches_op_chain(shape: tuple[int, int], dtype: mx.Dtype) -> None:
    experts, k = shape
    for seed in range(4):
        logits = replica(experts, k, seed, dtype, shared=False)
        bias = sigmoid_bias(experts, seed)
        indices, weights = softplus_topk(logits, bias, k, scale=SOFTPLUS_SCALE, norm=True)
        ref_indices, ref_weights = softplus_op_chain(logits, bias, k, norm=True)
        assert mx.array_equal(mx.sort(indices), mx.sort(ref_indices))
        ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
        order = sorted(theirs)
        assert relative_diff(
            mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
        ) < _weight_bound(mx.float32)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_softplus_exact_ties_go_to_the_higher_index(dtype: mx.Dtype) -> None:
    """Zero bias keeps `sqrt(softplus)` monotonic in the logits, so the tie group and
    the promised order are the softmax case's."""
    logits, k = tied_logits(dtype, shared=False)
    indices, _ = softplus_topk(logits, mx.zeros((logits.size,)), k, scale=SOFTPLUS_SCALE, norm=True)
    assert np.array(indices).tolist() == [0, 1, 2, 120, 99, 67, 35, 11]


def test_softplus_bias_selects_but_never_weighs() -> None:
    logits = replica(256, 6, 2, mx.bfloat16, shared=False)
    zero = mx.zeros((256,))
    base_idx, _ = softplus_topk(logits, zero, 6, scale=SOFTPLUS_SCALE, norm=True)
    outsider = next(e for e in range(256) if e not in set(np.array(base_idx).tolist()))
    boosted = mx.where(mx.arange(256) == outsider, 10.0, 0.0)
    indices, weights = softplus_topk(logits, boosted, 6, scale=SOFTPLUS_SCALE, norm=True)
    assert outsider in set(np.array(indices).tolist())
    ref_indices, ref_weights = softplus_op_chain(logits, boosted, 6, norm=True)
    ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
    order = sorted(theirs)
    assert relative_diff(
        mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
    ) < _weight_bound(mx.float32)


def test_softplus_without_norm_keeps_raw_scores() -> None:
    logits = replica(256, 6, 3, mx.bfloat16, shared=False)
    bias = sigmoid_bias(256, 3)
    indices, weights = softplus_topk(logits, bias, 6, scale=SOFTPLUS_SCALE, norm=False)
    ref_indices, ref_weights = softplus_op_chain(logits, bias, 6, norm=False)
    assert mx.array_equal(mx.sort(indices), mx.sort(ref_indices))
    ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
    order = sorted(theirs)
    assert relative_diff(
        mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
    ) < _weight_bound(mx.float32)


def test_softplus_gather_matches_op_chain() -> None:
    """The hash variant: experts given (duplicates included — a hash table may repeat),
    only the weights computed."""
    for seed in range(4):
        logits = replica(256, 6, seed, mx.bfloat16, shared=False)
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, 256, size=6)
        if seed == 0:
            picks[1] = picks[0]
        chosen = mx.array(picks.astype(np.uint32))
        weights = softplus_gather(logits, chosen, scale=SOFTPLUS_SCALE, norm=True)
        scores = np.array(mx.sqrt(nn.softplus(logits.astype(mx.float32))))
        ref = scores[picks]
        ref = ref / (ref.sum() + 1e-20) * SOFTPLUS_SCALE
        assert relative_diff(weights, mx.array(ref.astype(np.float32))) < _weight_bound(
            mx.float32
        )


_SOFTPLUS_MUTATIONS = {
    "resolve a lane's tie downwards": (
        "for (int i = (int)per_lane - 1; i >= 0; i--) {",
        "for (int i = 0; i < (int)per_lane; i++) {",
    ),
    "keep picking the same expert": (
        "if (winner == cand && slot >= 0) b[(uint)slot] = -INFINITY;",
        "if (false) b[(uint)slot] = -INFINITY;",
    ),
    "drop the renormalization": (
        "float w = NORM ? pick[lane] / (total + 1e-20f) : pick[lane];",
        "float w = pick[lane];",
    ),
    "ignore the bias in selection": (
        "b[i] = s[i] + B[lane + i * 32];",
        "b[i] = s[i];",
    ),
    "score without the sqrt": (
        "s[i] = metal::precise::sqrt(mlx_softplus(v));",
        "s[i] = mlx_softplus(v);",
    ),
}


def _softplus_route(
    source: str, name: str, logits: mx.array, bias: mx.array, k: int
) -> list[mx.array]:
    return metal_kernel(
        name=name,
        input_names=["L", "B", "SC"],
        output_names=["OI", "OW"],
        source=source,
        header=_SOFTPLUS_HEADER,
    )(
        inputs=[logits, bias, mx.array(SOFTPLUS_SCALE, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", logits.size), ("NORM", 1)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, mx.float32],
    )


@pytest.mark.parametrize("mutation", sorted(_SOFTPLUS_MUTATIONS))
def test_softplus_mutations_break_the_routing(mutation: str) -> None:
    old, new = _SOFTPLUS_MUTATIONS[mutation]
    source = _SOFTPLUS_SOURCE.replace(old, new)
    assert source != _SOFTPLUS_SOURCE
    logits, k = tied_logits(mx.float32, shared=False)
    bias = mx.where(mx.arange(logits.size) == logits.size - 1, 10.0, 0.0)
    ref_indices, ref_weights = softplus_op_chain(logits, bias, k, norm=True)
    broken = _softplus_route(
        source,
        f"moe_softplus_broken_{sorted(_SOFTPLUS_MUTATIONS).index(mutation)}",
        logits,
        bias,
        k,
    )
    picked_changed = not mx.array_equal(mx.sort(broken[0]), mx.sort(ref_indices))
    ours, theirs = _by_index(broken[0], broken[1]), _by_index(ref_indices, ref_weights)
    weights_changed = any(
        e not in theirs or abs(w - theirs[e]) > 1e-2 * abs(theirs[e]) for e, w in ours.items()
    )
    assert picked_changed or weights_changed


def test_softplus_gather_mutation_breaks_the_weights() -> None:
    source = _SOFTPLUS_GATHER_SOURCE.replace(
        "float w = NORM ? s / (total + 1e-20f) : s;", "float w = s;"
    )
    assert source != _SOFTPLUS_GATHER_SOURCE
    logits = replica(256, 6, 1, mx.float32, shared=False)
    chosen = mx.array(np.arange(6).astype(np.uint32))
    ref = softplus_gather(logits, chosen, scale=SOFTPLUS_SCALE, norm=True)
    (broken,) = metal_kernel(
        name="moe_softplus_gather_broken",
        input_names=["L", "C", "SC"],
        output_names=["OW"],
        source=source,
        header=_SOFTPLUS_HEADER,
    )(
        inputs=[logits, chosen, mx.array(SOFTPLUS_SCALE, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", 6), ("NORM", 1)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(6,)],
        output_dtypes=[mx.float32],
    )
    assert relative_diff(broken, ref) > 1e-2


def test_softplus_unmutated_replica_agrees_with_the_module() -> None:
    logits, k = tied_logits(mx.float32, shared=False)
    bias = sigmoid_bias(logits.size, 0)
    indices, weights = softplus_topk(logits, bias, k, scale=SOFTPLUS_SCALE, norm=True)
    intact = _softplus_route(_SOFTPLUS_SOURCE, "moe_softplus_intact", logits, bias, k)
    assert mx.array_equal(intact[0], indices)
    assert mx.array_equal(intact[1], weights)


# --- the delegator ------------------------------------------------------------------
# Resolution is the contract under test: every gate is an arithmetic claim, so a
# declaration the kernel does not compute has to fall through to `DefaultRoute` rather
# than be served a near-miss. The chain itself is covered above, per kernel.


def _gate(rows: int) -> mx.array:
    return mx.zeros((rows, 128))


def test_delegator_resolves_each_family() -> None:
    bias = mx.zeros((128,), dtype=mx.float32)
    wide = mx.zeros((256,), dtype=mx.float32)
    narrow = mx.zeros((32,), dtype=mx.float32)
    assert isinstance(Route(_gate(128), experts=128, k=8).strategy, SoftmaxTopkRoute)
    assert isinstance(
        Route(_gate(129), experts=128, k=8, shared=True).strategy, SoftmaxTopkRoute
    )
    assert isinstance(
        Route(_gate(128), experts=128, k=8, scoring="sigmoid", bias=bias, scale=2.5).strategy,
        SoftmaxTopkRoute,
    )
    assert isinstance(
        Route(
            _gate(128),
            experts=128,
            k=8,
            scoring="sqrt_softplus",
            bias=bias,
            norm_eps=1e-20,
            scale=2.5,
        ).strategy,
        SoftmaxTopkRoute,
    )
    assert isinstance(
        Route(
            _gate(128), experts=128, k=8, bias=bias, normalize=False, scale=6.0
        ).strategy,
        BiasTopkRoute,
    )
    assert isinstance(
        Route(_gate(256), experts=256, k=8, scoring="sigmoid", bias=wide).strategy,
        OrdinalRoute,
    )
    assert isinstance(
        Route(
            _gate(32), experts=32, k=4, scoring="sigmoid", bias=narrow, norm_eps=1e-6
        ).strategy,
        SigmoidTopkRoute,
    )


def test_delegator_falls_back_rather_than_substituting_arithmetic() -> None:
    bias = mx.zeros((128,), dtype=mx.float32)
    # The shape no kernel tiles.
    assert isinstance(Route(_gate(100), experts=100, k=8).strategy, DefaultRoute)
    # A softcap none of them applies.
    assert isinstance(
        Route(_gate(128), experts=128, k=8, softcap=30.0).strategy, DefaultRoute
    )
    # A scale the renormalizing softmax kernel would drop.
    assert isinstance(
        Route(_gate(128), experts=128, k=8, scale=2.5).strategy, DefaultRoute
    )
    # A selection bias the softmax kernel does not read.
    assert isinstance(
        Route(_gate(128), experts=128, k=8, bias=bias).strategy, DefaultRoute
    )
    # A `sqrt(softplus)` renormalization with the wrong epsilon.
    assert isinstance(
        Route(_gate(128), experts=128, k=8, scoring="sqrt_softplus", bias=bias).strategy,
        DefaultRoute,
    )
    # The renormalizing sigmoid kernel declines an unnormalized declaration; the wide
    # sigmoid family carries an explicit NORM switch, so it serves it instead.
    assert isinstance(
        Route(
            _gate(128), experts=128, k=8, scoring="sigmoid", bias=bias, normalize=False
        ).strategy,
        SigmoidWideRoute,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
def test_default_route_agrees_with_the_softmax_kernel(dtype: mx.Dtype, shared: bool) -> None:
    """The two ends of the delegator on one declaration: same pick, same weights within
    the family's bound."""
    experts, k = 128, 8
    routing = Routing(experts=experts, k=k, shared=shared)
    gate = mx.zeros((experts + (1 if shared else 0), 4), dtype=dtype)
    fused = SoftmaxTopkRoute.build(gate, routing=routing)
    eager = DefaultRoute.build(gate, routing=routing)
    assert fused is not None
    for seed in range(4):
        logits = replica(experts, k, seed, dtype, shared=shared)
        row = mx.zeros((4,), dtype=dtype)
        indices, weights = fused(row, logits=logits)
        ref_indices, ref_weights = eager(row, logits=logits)
        assert mx.array_equal(mx.sort(indices), mx.sort(ref_indices))
        ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
        order = sorted(theirs)
        assert relative_diff(
            mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
        ) < _weight_bound(dtype)


# --- wide sigmoid routing (nemotron_h): sigmoid+bias selection past 32 experts ---


@pytest.mark.parametrize("normalize", [True, False])
def test_sigmoid_wide_matches_the_op_chain(normalize: bool) -> None:
    """The DeepSeek-style sigmoid pick at 128 experts against the model's own ops:
    same selection set, weights within the documented renorm-order bound."""
    experts, k, hidden = 128, 6, 64
    mx.random.seed(31)
    gate = mx.random.normal((experts, hidden)).astype(mx.bfloat16)
    bias = mx.random.normal((experts,)).astype(mx.float32)
    routing = Routing(
        experts=experts, k=k, scoring="sigmoid", bias=bias,
        normalize=normalize, norm_eps=1e-20, scale=2.5,
    )
    wide = SigmoidWideRoute.build(gate, routing=routing)
    assert wide is not None
    for seed in range(4):
        mx.random.seed(100 + seed)
        row = mx.random.normal((hidden,)).astype(mx.bfloat16)
        logits = row @ gate.T
        indices, weights = wide(row, logits=logits)

        scores = mx.sigmoid(logits.astype(mx.float32))
        selector = scores + bias
        chosen = mx.argpartition(-selector, kth=k - 1, axis=-1)[:k]
        wanted = mx.take_along_axis(scores, chosen, axis=-1)
        if normalize:
            wanted = wanted / (wanted.sum() + 1e-20)
        wanted = wanted * 2.5
        assert mx.array_equal(mx.sort(indices), mx.sort(chosen.astype(mx.uint32)))
        ours = _by_index(indices, weights)
        theirs = _by_index(chosen.astype(mx.uint32), wanted)
        order = sorted(theirs)
        assert relative_diff(
            mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
        ) < 1e-6


def test_sigmoid_wide_exact_ties_go_to_the_higher_index() -> None:
    """Equal logits and a zero bias make the biased scores exactly equal; the cut falls
    inside the tie group, so the rule alone decides which indices survive."""
    experts, k, hidden = 128, 8, 64
    logits, _ = tied_logits(mx.float32, shared=False)
    routing = Routing(
        experts=experts, k=k, scoring="sigmoid",
        bias=mx.zeros((experts,), dtype=mx.float32),
        normalize=True, norm_eps=1e-20, scale=2.5,
    )
    wide = SigmoidWideRoute.build(mx.zeros((experts, hidden)), routing=routing)
    assert wide is not None
    indices, _ = wide(mx.zeros((hidden,)), logits=logits)
    assert indices.tolist() == [0, 1, 2, 120, 99, 67, 35, 11]


def test_sigmoid_wide_rows_matches_per_row() -> None:
    """T rows through the rows dispatch equal the single-row kernel row for row."""
    experts, k, hidden = 128, 6, 64
    mx.random.seed(41)
    gate = mx.random.normal((experts, hidden)).astype(mx.bfloat16)
    bias = mx.random.normal((experts,)).astype(mx.float32)
    routing = Routing(
        experts=experts, k=k, scoring="sigmoid", bias=bias,
        normalize=True, norm_eps=1e-20, scale=2.5,
    )
    wide = SigmoidWideRoute.build(gate, routing=routing)
    assert wide is not None
    rows = mx.random.normal((3, hidden)).astype(mx.bfloat16)
    logits = mx.stack([row @ gate.T for row in rows])
    chosen, weights = wide.rows(logits)
    for t in range(3):
        one_c, one_w = wide(rows[t], logits=logits[t])
        assert mx.array_equal(chosen[t], one_c).item(), t
        assert mx.array_equal(weights[t], one_w).item(), t

