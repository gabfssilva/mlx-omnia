"""Parity for the fused residual-join strategies (`core.kernels.add_norm`).

The replica is the complete step the kernels replace at the qwen3_moe / qwen3_5
residual join — `attended = x + branch; normed = rms_norm(attended, w, eps)` — at
checkpoint shapes (hidden 2048/4096, plus 2880 for the VPT=5 tiling) and
checkpoint-like magnitudes: a residual stream that has grown along the trunk
(std ~30, a few outlier channels an order of magnitude above), a branch an order of
magnitude smaller, and RMSNorm weights around 1. In N(0,1) the sum of squares lands
where one bf16 ulp separates nothing.

The bf16 bound is not invented: both paths are compared against the same step done
in fp32, and the kernel is required to be no further from it than the stock op
chain is (the floor is what bf16 rounding costs, measured on the spot).
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import relative_diff

from sideros.core.kernels.add_norm import (
    AddRmsNorm,
    DefaultAddRmsNorm,
    FusedAddRmsNorm,
    RowsAddRmsNorm,
)
from sideros.core.kernels.add_norm import fused as fused_module
from sideros.core.kernels.add_norm import rows as rows_module

EPS = 1e-6
HIDDENS = [1024, 2048, 4096, 2880]


def replica(hidden: int, dtype: mx.Dtype, seed: int = 0) -> tuple[mx.array, mx.array, mx.array]:
    """Residual stream, branch output and norm weight at checkpoint-like magnitudes."""
    mx.random.seed(seed)
    x = mx.random.normal((hidden,)) * 30.0
    outliers = mx.random.normal((hidden,)) * 300.0 * (mx.random.uniform(shape=(hidden,)) > 0.99)
    x = x + outliers
    branch = mx.random.normal((hidden,)) * 3.0
    weight = 1.0 + mx.random.normal((hidden,)) * 0.2
    return x.astype(dtype), branch.astype(dtype), weight.astype(dtype)


def leaf_of(weight: mx.array) -> nn.RMSNorm:
    norm = nn.RMSNorm(weight.size, eps=EPS)
    norm.weight = weight
    return norm


def op_chain(
    x: mx.array, branch: mx.array, weight: mx.array
) -> tuple[mx.array, mx.array]:
    added = x + branch
    return added, mx.fast.rms_norm(added, weight, EPS)


@pytest.mark.parametrize("hidden", HIDDENS)
def test_applies_predicate(hidden: int) -> None:
    assert fused_module.applies(hidden)
    assert not fused_module.applies(100)
    assert not fused_module.applies(16384)


def test_rejects_hidden_over_the_threadgroup_limit() -> None:
    """8192 tiles as 2048 threads (VPT=4) and Metal caps a threadgroup at 1024, so the
    dispatch would fail; 5120 (the 27B's) is the same story at 1280."""
    assert not fused_module.applies(8192)
    assert not fused_module.applies(5120)
    assert fused_module.applies(4096)


def test_accepted_hidden_fits_one_threadgroup() -> None:
    """Whatever the predicate accepts, the dispatch asks for `hidden // VPT` threads in a
    single threadgroup — never more than the 1024 the device allows."""
    accepted = [hidden for hidden in range(32, 8193, 32) if fused_module.applies(hidden)]
    assert 4096 in accepted
    for hidden in accepted:
        vpt = 4 if hidden % 128 == 0 else 5
        assert hidden % vpt == 0
        assert 0 < hidden // vpt <= 1024


def test_delegator_resolves_by_declaration() -> None:
    """One token over a tiling hidden takes the fused kernel; an undeclared token count
    takes the row-walking one; a hidden neither tiles falls through to the two ops."""
    weight = replica(2048, mx.float32)[2]
    assert isinstance(AddRmsNorm(leaf_of(weight), tokens=1).strategy, FusedAddRmsNorm)
    assert isinstance(AddRmsNorm(leaf_of(weight)).strategy, RowsAddRmsNorm)
    wide = replica(8192, mx.float32)[2]
    assert isinstance(AddRmsNorm(leaf_of(wide), tokens=1).strategy, DefaultAddRmsNorm)


@pytest.mark.parametrize("hidden", HIDDENS)
def test_fp32_matches_op_chain(hidden: int) -> None:
    x, branch, weight = replica(hidden, mx.float32)
    added, normed = AddRmsNorm(leaf_of(weight), tokens=1)(x, branch)
    ref_added, ref_normed = op_chain(x, branch, weight)
    assert relative_diff(added, ref_added) == 0.0
    assert relative_diff(normed, ref_normed) < 1e-6


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("hidden", HIDDENS)
def test_narrow_dtype_within_rounding_floor(hidden: int, dtype: mx.Dtype) -> None:
    x, branch, weight = replica(hidden, dtype)
    added, normed = AddRmsNorm(leaf_of(weight), tokens=1)(x, branch)
    ref_added, ref_normed = op_chain(x, branch, weight)
    exact_normed = op_chain(
        x.astype(mx.float32), branch.astype(mx.float32), weight.astype(mx.float32)
    )[1]
    floor = relative_diff(ref_normed, exact_normed)
    assert floor > 0.0
    assert relative_diff(added, ref_added) == 0.0
    assert relative_diff(normed, exact_normed) <= floor


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_keeps_token_shape(dtype: mx.Dtype) -> None:
    x, branch, weight = replica(2048, dtype)
    token, branch_token = x.reshape(1, 1, 2048), branch.reshape(1, 1, 2048)
    added, normed = AddRmsNorm(leaf_of(weight), tokens=1)(token, branch_token)
    assert added.shape == (1, 1, 2048) and normed.shape == (1, 1, 2048)
    ref_added, ref_normed = op_chain(token, branch_token, weight)
    exact_normed = op_chain(
        token.astype(mx.float32), branch_token.astype(mx.float32), weight.astype(mx.float32)
    )[1]
    floor = max(relative_diff(ref_normed, exact_normed), 1e-6)
    assert relative_diff(added, ref_added) == 0.0
    assert relative_diff(normed, exact_normed) <= floor


def test_eps_reaches_the_kernel() -> None:
    """A tiny stream is where eps shows: the kernel must not carry a frozen one."""
    x, branch, weight = replica(2048, mx.float32)
    x, branch = x * 1e-3, branch * 1e-3
    leaf = leaf_of(weight)
    leaf.eps = 0.5
    _, normed = AddRmsNorm(leaf, tokens=1)(x, branch)
    _, reference = op_chain(x, branch, weight)
    assert relative_diff(normed, reference) > 1e-2
    added = x + branch
    assert relative_diff(normed, mx.fast.rms_norm(added, weight, 0.5)) < 1e-6


@pytest.mark.parametrize("hidden", [1024, 2048, 4096])
def test_rows_matches_op_chain_over_many_rows(hidden: int) -> None:
    """The row-walking strategy is the other numerical path: it rounds the normed value
    to T before the norm weight multiplies it, so it is pinned against fp32 directly."""
    assert rows_module.applies(hidden)
    x, branch, weight = replica(hidden, mx.float32)
    block = mx.stack([x, x * 0.5, x * 2.0])
    branch_block = mx.stack([branch, branch * 2.0, branch * 0.25])
    strategy = AddRmsNorm(leaf_of(weight)).strategy
    assert isinstance(strategy, RowsAddRmsNorm)
    added, normed = strategy(block, branch_block)
    ref_added, ref_normed = op_chain(block, branch_block, weight)
    assert added.shape == block.shape and normed.shape == block.shape
    assert relative_diff(added, ref_added) == 0.0
    assert relative_diff(normed, ref_normed) < 1e-6


def test_rows_rejects_non_tiling_hidden() -> None:
    assert not rows_module.applies(2880)
    assert not rows_module.applies(8192)
