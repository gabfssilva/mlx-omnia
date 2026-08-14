# pyright: basic
"""softmax_bias_topk: softmax → bias-to-scores → top-k (no renorm) in one dispatch.

The replica is the routing step the kernel replaces — softmax(logits, precise=True),
add ``e_score_correction_bias`` to the scores, top-k on the biased scores (ties
to the higher index), weights from the raw scores x ``routed_scaling_factor``.

The Longcat Flash router differs from both existing kernels:
- ``softmax_topk`` renormalizes and takes no bias.
- ``sigmoid_topk`` is a different family (sigmoid, bias on logits).
- This kernel: softmax, bias on scores, no renorm, weights = raw x scale.

Selection is bit-exact (asserted with ``mx.array_equal``). Weights are not
bit-exact (the kernel rounds in its own order) and are bounded by a handful of
T ulps + fp32 slack, the same bound ``test_kernel_moe_route`` uses.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia.engine.core.kernels.route.bias_topk import (
    _SOURCE,
    softmax_bias_topk,
    softmax_bias_topk_applies,
)
from mlx_omnia.engine.core.mxcompat import metal_kernel, softmax
from tests.conftest import relative_diff

SHAPES = [(32, 4), (128, 8), (256, 8), (384, 12)]
SCALE = 6.0


def _ulps(dtype: mx.Dtype, count: int) -> float:
    return count * (2.0**-23 if dtype == mx.float32 else 2.0**-8)


def _weight_bound(dtype: mx.Dtype) -> float:
    return _ulps(dtype, 8) + 32 * 2.0**-23


def replica(experts: int, k: int, seed: int, dtype: mx.Dtype) -> mx.array:
    """Random router logits whose top-k stands clear of the rest by two logits."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(experts)
    logits[np.argsort(logits, kind="stable")[-k:]] += 2.0
    return mx.array(logits.astype(np.float32)).astype(dtype)


def bias_vector(experts: int, seed: int) -> mx.array:
    return mx.array(
        np.random.default_rng(seed + 100).standard_normal(experts) * 0.1
    ).astype(mx.float32)


def op_chain(
    logits: mx.array, bias: mx.array, k: int, scale: float, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """The picked indices in the kernel's promised order and the raw-score x scale
    weights (no renorm)."""
    scores = np.array(softmax(logits, axis=-1, precise=True).astype(mx.float32))
    scores_t = mx.array(scores.astype(np.float32)).astype(dtype)
    scores_t_np = np.array(scores_t.astype(mx.float32))
    corrected = scores_t_np + np.array(bias)
    chosen = np.argsort(corrected, kind="stable")[::-1][:k]
    weights = np.array(scores_t)[chosen] * scale
    return mx.array(chosen.astype(np.uint32)), mx.array(weights.astype(np.float32))


def _by_index(indices: mx.array, weights: mx.array) -> dict[int, float]:
    picked = np.array(indices).tolist()
    values = np.array(weights.astype(mx.float32)).tolist()
    return dict(zip(picked, values, strict=True))


def test_applies_predicate() -> None:
    for experts, k in SHAPES:
        assert softmax_bias_topk_applies(experts, k)
    assert not softmax_bias_topk_applies(100, 8)
    assert not softmax_bias_topk_applies(16, 8)
    assert not softmax_bias_topk_applies(128, 33)
    assert not softmax_bias_topk_applies(128, 0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_op_chain(shape: tuple[int, int], dtype: mx.Dtype) -> None:
    experts, k = shape
    for seed in range(4):
        logits = replica(experts, k, seed, dtype)
        bias = bias_vector(experts, seed)
        indices, weights = softmax_bias_topk(logits, bias, k, scale=SCALE)
        ref_indices, ref_weights = op_chain(logits, bias, k, SCALE, dtype)
        assert indices.shape == (k,)
        assert mx.array_equal(mx.sort(indices), mx.sort(ref_indices))

        ours = _by_index(indices, weights)
        theirs = _by_index(ref_indices, ref_weights)
        order = sorted(theirs)
        assert relative_diff(
            mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
        ) < _weight_bound(dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_no_renormalization(dtype: mx.Dtype) -> None:
    """The weights are raw softmax scores x scale, NOT renormalized — their sum
    is ``scale x (sum of k raw scores)``, which is ``scale`` only when the kept
    scores happen to sum to 1."""
    experts, k = 256, 8
    logits = replica(experts, k, 3, dtype)
    bias = bias_vector(experts, 3)
    _, weights = softmax_bias_topk(logits, bias, k, scale=SCALE)
    total = float(np.array(weights.astype(mx.float32)).sum())
    # If renormalized, the sum would be ~SCALE (6.0). The raw sum of 8 softmax
    # scores is well below 1, so the total is well below SCALE.
    assert total < SCALE * 0.9


def tied_logits(dtype: mx.Dtype) -> tuple[mx.array, int]:
    """128 strictly decreasing logits with one group of six forced exactly equal.

    The group sits at rank 4 and spans six ranks, so the k=8 cut falls inside it:
    five of the six are kept and which one is dropped is the tie rule's alone.
    Indices 3/35/67/99 are the same lane (four slots of one thread), 11 and 120
    are two other lanes — the rule has to hold both inside a lane's registers
    and across the simd_max.
    """
    values = np.arange(128, dtype=np.float64) * -0.5
    values[[3, 11, 35, 67, 99, 120]] = values[3]
    return mx.array(values.astype(np.float32)).astype(dtype), 8


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_exact_ties_go_to_the_higher_index(dtype: mx.Dtype) -> None:
    logits, k = tied_logits(dtype)
    zero_bias = mx.zeros((logits.size,))
    indices, _ = softmax_bias_topk(logits, zero_bias, k, scale=SCALE)
    ref_indices, _ = op_chain(logits, zero_bias, k, SCALE, dtype)
    expected = [0, 1, 2, 120, 99, 67, 35, 11]
    assert np.array(ref_indices).tolist() == expected
    assert mx.array_equal(indices, ref_indices)


def test_bias_selects_but_never_weighs() -> None:
    """Boosting one outsider's bias must pull it in, while its weight still comes
    from the unbiased score — the biased value must not leak into the weight."""
    logits = replica(256, 8, 2, mx.bfloat16)
    zero = mx.zeros((256,))
    base_idx, _ = softmax_bias_topk(logits, zero, 8, scale=SCALE)
    outsider = next(e for e in range(256) if e not in set(np.array(base_idx).tolist()))
    boosted = mx.where(mx.arange(256) == outsider, 10.0, 0.0)
    indices, weights = softmax_bias_topk(logits, boosted, 8, scale=SCALE)
    assert outsider in set(np.array(indices).tolist())
    ref_indices, ref_weights = op_chain(logits, boosted, 8, SCALE, mx.bfloat16)
    ours, theirs = _by_index(indices, weights), _by_index(ref_indices, ref_weights)
    order = sorted(theirs)
    assert relative_diff(
        mx.array([ours[i] for i in order]), mx.array([theirs[i] for i in order])
    ) < _weight_bound(mx.bfloat16)


_MUTATIONS = {
    "resolve a lane's tie downwards": (
        "for (int i = (int)per_lane - 1; i >= 0; i--) {",
        "for (int i = 0; i < (int)per_lane; i++) {",
    ),
    "keep picking the same expert": (
        "if (winner == cand && slot >= 0) b[(uint)slot] = -INFINITY;",
        "if (false) b[(uint)slot] = -INFINITY;",
    ),
    "add bias to logits instead of scores": (
        "b[i] = (float)p[i] + (float)B[lane + i * 32];",
        "p[i] = (float)p[i] + (float)B[lane + i * 32];\n        b[i] = p[i];",
    ),
    "renormalize the weights": (
        "T w = (T)pick[lane];\n        OW[lane] = (T)((float)w * SC);",
        "T w = (T)pick[lane];\n        float total = 0.0f;\n"
        "        for (int j = 0; j < TOPK; j++) total += pick[j];\n"
        "        OW[lane] = (T)((float)w / total * SC);",
    ),
}


def _route(
    source: str, name: str, logits: mx.array, bias: mx.array, k: int
) -> list[mx.array]:
    return metal_kernel(
        name=name, input_names=["L", "B", "SC"], output_names=["OI", "OW"], source=source
    )(
        inputs=[logits, bias.astype(mx.float32), mx.array(SCALE, dtype=mx.float32)],
        template=[("T", logits.dtype), ("TOPK", k), ("EXPERTS", logits.size)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.uint32, logits.dtype],
    )


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_mutations_break_the_routing(mutation: str) -> None:
    """The tie case exercises the pick loop; the boosted-outsider bias makes the
    bias term decisive, so dropping it changes the selection."""
    old, new = _MUTATIONS[mutation]
    source = _SOURCE.replace(old, new)
    assert source != _SOURCE
    logits, k = tied_logits(mx.float32)
    bias = mx.where(mx.arange(logits.size) == logits.size - 1, 10.0, 0.0)
    ref_indices, ref_weights = op_chain(logits, bias, k, SCALE, mx.float32)
    broken = _route(
        source,
        f"softmax_bias_topk_broken_{sorted(_MUTATIONS).index(mutation)}",
        logits,
        bias,
        k,
    )
    picked_changed = not mx.array_equal(mx.sort(broken[0]), mx.sort(ref_indices))
    ours, theirs = _by_index(broken[0], broken[1]), _by_index(ref_indices, ref_weights)
    weights_changed = any(
        e not in theirs or abs(w - theirs[e]) > 1e-2 * abs(theirs[e])
        for e, w in ours.items()
    )
    assert picked_changed or weights_changed


def test_unmutated_replica_of_the_dispatch_agrees_with_the_module() -> None:
    logits, k = tied_logits(mx.float32)
    bias = bias_vector(logits.size, 0)
    indices, weights = softmax_bias_topk(logits, bias, k, scale=SCALE)
    intact = _route(_SOURCE, "softmax_bias_topk_intact", logits, bias, k)
    assert mx.array_equal(intact[0], indices)
    assert mx.array_equal(intact[1], weights)
