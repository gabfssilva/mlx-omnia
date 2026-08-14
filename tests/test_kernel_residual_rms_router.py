"""Residual add + rms_norm, and the fused router gemv, against the stock op chain.

The chain the kernels replace is `residual + branch`, `mx.fast.rms_norm`, and — for the
fused one — a matmul against the router rows. What is exact and what is not, kept apart:

* `residual + branch` **is** bit-exact and is asserted with `mx.array_equal`: the kernel
  rounds the sum to `T` exactly where the materialized tensor between the two ops would;
* the normed row and the logits are not. The kernel rounds the normed value to `T` before
  the norm weight multiplies it (`mx.fast.rms_norm` keeps the product in fp32), reduces
  with `metal::precise::rsqrt`, and accumulates the gemv per lane and then through a
  shuffle tree. The bound is the dtype's own rounding over a `hidden`-long dot product;
* the sort keys are an *ordering*, so they are asserted as one: the permutation the keys
  induce has to be the reference's stable descending order of `sigmoid(logit) + bias`,
  exactly, and `router_tournament` reading the same row has to pick its first `k`.

`rows_per_group` is a tiling knob and nothing else, so all seven settings are asserted
bitwise identical to each other on all four outputs — that is the property that makes it
safe to sweep.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia.engine.core.kernels.add_norm.rows import _NORM_SOURCE, RowsAddRmsNorm
from mlx_omnia.engine.core.kernels.add_norm.rows import applies as residual_rms_norm_applies
from mlx_omnia.engine.core.kernels.route.ordinal import ORDINAL_HEADER, router_tournament
from mlx_omnia.engine.core.kernels.route.residual import (
    _router_source,
    residual_rms_router,
    residual_rms_router_applies,
)
from mlx_omnia.engine.core.mxcompat import metal_kernel
from tests.conftest import relative_diff

EPS = 1e-6
ROWS_PER_GROUP = [1, 2, 4, 8, 16, 32, 64]
SHAPES = [(2048, 256), (512, 64)]


def residual_rms_norm(
    residual: mx.array, branch: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    return RowsAddRmsNorm(weight, eps)(residual, branch)


def _ulps(dtype: mx.Dtype, count: int) -> float:
    return count * (2.0**-23 if dtype == mx.float32 else 2.0**-8)


def _bound(dtype: mx.Dtype) -> float:
    """fp32 keeps the 1e-5 the house asks of a template; bf16 pays four of its own ulps,
    which is the rounding of the normed row carried through a `hidden`-long dot."""
    return 1e-5 if dtype == mx.float32 else _ulps(dtype, 4)


def _row(shape: tuple[int, ...], seed: int, dtype: mx.Dtype, scale: float = 1.0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """The kernel's own branchless form, in fp64 on purpose: the reference is the
    mathematical sigmoid, not a replay of Metal's `exp`."""
    y = 1.0 / (1.0 + np.exp(np.abs(x)))
    return np.where(x < 0.0, y, 1.0 - y)


def _summed_ref(residual: mx.array, branch: mx.array) -> mx.array:
    return (residual.astype(mx.float32) + branch.astype(mx.float32)).astype(residual.dtype)


def _normalized_ref(summed: mx.array, weight: mx.array) -> mx.array:
    return mx.fast.rms_norm(summed.astype(mx.float32), weight.astype(mx.float32), EPS)


def _logits_ref(normalized: mx.array, router_weight: mx.array) -> mx.array:
    return normalized.astype(mx.float32) @ router_weight.astype(mx.float32).T


def _selection(scores: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Stable descending order of `score + bias` — ties to the lower index, which is what
    ascending order of the negated key means."""
    return np.argsort(-(scores + bias), kind="stable")


def test_applies_predicates() -> None:
    assert residual_rms_norm_applies(2048)
    assert residual_rms_norm_applies(512)
    assert not residual_rms_norm_applies(2000)
    assert not residual_rms_norm_applies(8192)

    assert residual_rms_router_applies(2048, 256, 8)
    for rows_per_group in ROWS_PER_GROUP:
        assert residual_rms_router_applies(2048, 256, rows_per_group)
        assert residual_rms_router_applies(512, 64, rows_per_group)
    assert not residual_rms_router_applies(2048, 256, 3)
    assert not residual_rms_router_applies(2048, 250, 8)
    assert not residual_rms_router_applies(2048, 256, 24)
    assert not residual_rms_router_applies(384, 256, 1)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("hidden", [2048, 512])
def test_residual_rms_norm_matches_the_op_chain(hidden: int, dtype: mx.Dtype) -> None:
    residual, branch = _row((3, hidden), 0, dtype), _row((3, hidden), 1, dtype)
    weight = _row((hidden,), 2, dtype, scale=0.1)
    summed, normalized = residual_rms_norm(residual, branch, weight, EPS)

    assert summed.shape == residual.shape
    assert mx.array_equal(summed, _summed_ref(residual, branch))
    assert relative_diff(normalized, _normalized_ref(summed, weight)) < _bound(dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_router_fusion_matches_the_op_chain(shape: tuple[int, int], dtype: mx.Dtype) -> None:
    hidden, experts = shape
    residual, branch = _row((1, 1, hidden), 0, dtype), _row((1, 1, hidden), 1, dtype)
    weight = _row((hidden,), 2, dtype, scale=0.1)
    router_weight = _row((experts, hidden), 3, dtype, scale=hidden**-0.5)
    bias = mx.array(np.random.default_rng(4).standard_normal(experts).astype(np.float32) * 0.1)

    summed, normalized, logits, keys = residual_rms_router(
        residual, branch, weight, router_weight, bias, eps=EPS
    )
    assert summed.shape == residual.shape == normalized.shape
    assert logits.shape == (1, 1, experts) == keys.shape
    assert keys.dtype == mx.uint32

    assert mx.array_equal(summed, _summed_ref(residual, branch))
    reference_normed = _normalized_ref(summed, weight)
    assert relative_diff(normalized, reference_normed) < _bound(dtype)
    assert relative_diff(logits, _logits_ref(reference_normed, router_weight)) < _bound(dtype)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_sort_keys_order_the_experts_the_way_the_reference_does(
    shape: tuple[int, int], dtype: mx.Dtype
) -> None:
    """The keys are compared as a permutation, not as bits: they are built from the
    kernel's own `metal::exp`, so bit equality against numpy would be testing Metal's
    exponential.

    The logits do not depend on the bias, so one pass with a zero bias yields the scores
    the second pass's bias is built from: `score + bias` lands on a randomly permuted grid
    four apart, straddling zero so the ordinal's sign branch is exercised on both sides,
    and every comparison the ordering depends on sits far clear of the difference between
    the kernel's fp32 sigmoid and the reference's fp64 one.
    """
    hidden, experts = shape
    residual, branch = _row((hidden,), 5, dtype), _row((hidden,), 6, dtype)
    weight = _row((hidden,), 7, dtype, scale=0.1)
    router_weight = _row((experts, hidden), 8, dtype, scale=hidden**-0.5)

    zero = mx.zeros((experts,))
    _, _, logits, _ = residual_rms_router(residual, branch, weight, router_weight, zero, eps=EPS)
    scores = _stable_sigmoid(np.array(logits.astype(mx.float32), dtype=np.float64))
    grid = np.empty(experts)
    grid[np.random.default_rng(9).permutation(experts)] = (np.arange(experts) - experts / 2) * 4.0
    bias = mx.array((grid - scores).astype(np.float32))

    _, _, again, keys = residual_rms_router(
        residual, branch, weight, router_weight, bias, eps=EPS
    )
    assert mx.array_equal(again, logits)
    expected = _selection(scores, np.array(bias, dtype=np.float64))
    assert np.argsort(np.array(keys), kind="stable").tolist() == expected.tolist()

    if experts == 256:
        picked, _ = router_tournament(logits.astype(mx.float32), bias, 8)
        assert np.array(picked).tolist() == expected[:8].tolist()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_rows_per_group_is_a_pure_tiling_knob(shape: tuple[int, int], dtype: mx.Dtype) -> None:
    hidden, experts = shape
    residual, branch = _row((hidden,), 10, dtype), _row((hidden,), 11, dtype)
    weight = _row((hidden,), 12, dtype, scale=0.1)
    router_weight = _row((experts, hidden), 13, dtype, scale=hidden**-0.5)
    bias = mx.array(np.random.default_rng(14).standard_normal(experts).astype(np.float32) * 0.1)

    reference = residual_rms_router(
        residual, branch, weight, router_weight, bias, eps=EPS, rows_per_group=1
    )
    for rows_per_group in ROWS_PER_GROUP[1:]:
        got = residual_rms_router(
            residual, branch, weight, router_weight, bias, eps=EPS, rows_per_group=rows_per_group
        )
        for ours, theirs in zip(got, reference, strict=True):
            assert mx.array_equal(ours, theirs), rows_per_group


_NORM_MUTATIONS = {
    "drop the branch from the residual sum": (
        "T value = T(residual[base + i] + branch[base + i]);",
        "T value = T(residual[base + i]);",
    ),
    "drop the norm weight": (
        "    normalized[base + i] =\n        weight[lid * n_reads + i] *\n"
        "        T(float(values[i]) * inv_mean);",
        "    normalized[base + i] = T(float(values[i]) * inv_mean);",
    ),
    "drop the inverse rms": (
        "        T(float(values[i]) * inv_mean);",
        "        T(float(values[i]));",
    ),
}


@pytest.mark.parametrize("mutation", sorted(_NORM_MUTATIONS))
def test_norm_mutations_break_the_output(mutation: str) -> None:
    old, new = _NORM_MUTATIONS[mutation]
    source = _NORM_SOURCE.replace(old, new)
    assert source != _NORM_SOURCE
    hidden = 512
    residual, branch = _row((1, hidden), 0, mx.float32), _row((1, hidden), 1, mx.float32)
    weight = _row((hidden,), 2, mx.float32, scale=0.1)
    summed, normalized = residual_rms_norm(residual, branch, weight, EPS)
    broken = metal_kernel(
        name=f"residual_rms_norm_broken_{sorted(_NORM_MUTATIONS).index(mutation)}",
        input_names=["residual", "branch", "weight", "eps"],
        output_names=["summed", "normalized"],
        source=source,
    )(
        inputs=[residual, branch, weight, mx.array(EPS, dtype=mx.float32)],
        template=[("T", mx.float32), ("AXIS", hidden)],
        grid=(hidden // 4, 1, 1),
        threadgroup=(hidden // 4, 1, 1),
        output_shapes=[(1, hidden), (1, hidden)],
        output_dtypes=[mx.float32, mx.float32],
    )
    assert (
        relative_diff(broken[0], summed) > 1e-2 or relative_diff(broken[1], normalized) > 1e-2
    )


_ROUTER_MUTATIONS = {
    "regroup the shuffle reduction": (
        "for (ushort delta = 16; delta >= 1; delta >>= 1) {",
        "for (ushort delta = 16; delta >= 2; delta >>= 1) {",
    ),
    "sort the key the wrong way round": (
        "        router_keys[router_row + r] = router_key_ordinal(\n"
        "            -(score + float(correction_bias[router_row + r])));",
        "        router_keys[router_row + r] = router_key_ordinal(\n"
        "            (score + float(correction_bias[router_row + r])));",
    ),
    "ignore the correction bias": (
        "            -(score + float(correction_bias[router_row + r])));",
        "            -(score));",
    ),
}


@pytest.mark.parametrize("mutation", sorted(_ROUTER_MUTATIONS))
def test_router_mutations_break_the_logits_or_the_keys(mutation: str) -> None:
    hidden, experts, rows_per_group = 512, 64, 4
    old, new = _ROUTER_MUTATIONS[mutation]
    source = _router_source(hidden, rows_per_group).replace(old, new)
    assert source != _router_source(hidden, rows_per_group)

    residual, branch = _row((hidden,), 20, mx.float32), _row((hidden,), 21, mx.float32)
    weight = _row((hidden,), 22, mx.float32, scale=0.1)
    router_weight = _row((experts, hidden), 23, mx.float32, scale=hidden**-0.5)
    bias = mx.array(np.random.default_rng(24).standard_normal(experts).astype(np.float32))
    _, _, logits, keys = residual_rms_router(
        residual, branch, weight, router_weight, bias, eps=EPS, rows_per_group=rows_per_group
    )
    broken = metal_kernel(
        name=f"residual_rms_router_broken_{sorted(_ROUTER_MUTATIONS).index(mutation)}",
        input_names=["residual", "branch", "weight", "router_weight", "correction_bias", "eps"],
        output_names=["summed", "normalized", "router_logits", "router_keys"],
        source=source,
        header=ORDINAL_HEADER,
    )(
        inputs=[residual, branch, weight, router_weight, bias, mx.array(EPS, dtype=mx.float32)],
        template=[("T", mx.float32)],
        grid=((experts // rows_per_group) * (hidden // 4), 1, 1),
        threadgroup=(hidden // 4, 1, 1),
        output_shapes=[(hidden,), (hidden,), (experts,), (experts,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.uint32],
    )
    assert relative_diff(broken[2], logits) > 1e-3 or not mx.array_equal(broken[3], keys)


def test_unmutated_replica_of_the_dispatch_agrees_with_the_module() -> None:
    """The mutation dispatches above are only evidence if the untouched source, launched
    the same way by hand, reproduces what the module returns."""
    hidden, experts, rows_per_group = 512, 64, 4
    residual, branch = _row((hidden,), 20, mx.float32), _row((hidden,), 21, mx.float32)
    weight = _row((hidden,), 22, mx.float32, scale=0.1)
    router_weight = _row((experts, hidden), 23, mx.float32, scale=hidden**-0.5)
    bias = mx.array(np.random.default_rng(24).standard_normal(experts).astype(np.float32))
    expected = residual_rms_router(
        residual, branch, weight, router_weight, bias, eps=EPS, rows_per_group=rows_per_group
    )
    intact = metal_kernel(
        name="residual_rms_router_intact",
        input_names=["residual", "branch", "weight", "router_weight", "correction_bias", "eps"],
        output_names=["summed", "normalized", "router_logits", "router_keys"],
        source=_router_source(hidden, rows_per_group),
        header=ORDINAL_HEADER,
    )(
        inputs=[residual, branch, weight, router_weight, bias, mx.array(EPS, dtype=mx.float32)],
        template=[("T", mx.float32)],
        grid=((experts // rows_per_group) * (hidden // 4), 1, 1),
        threadgroup=(hidden // 4, 1, 1),
        output_shapes=[(hidden,), (hidden,), (experts,), (experts,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.uint32],
    )
    for ours, theirs in zip(intact, expected, strict=True):
        assert mx.array_equal(ours, theirs)
