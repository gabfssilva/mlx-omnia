"""conv_mix parity: the fused T=1 gated short-conv step against the model's op path.

The target is bit-exactness, not a tolerance. With the projections held exact (a
one-hot in_proj, whose gemv gives the same result in any reduction order) the fused
tail must reproduce the op chain's every rounding boundary — in fp32 and in bf16.

Against the real LFM2.5-8B-A1B layer-0 weights the only remaining difference is the
gemv's reduction order, which in bf16 rounds away entirely (measured: exact) and in
fp32 stays under the projection's own fp32-vs-fp64 floor, measured in the test.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia.engine.core.kernels.conv_mix import ConvMix, DefaultConvMix, FusedConvMix
from mlx_omnia.engine.core.kernels.conv_mix import fused as cm
from mlx_omnia.engine.core.mxcompat import metal_kernel
from tests.conftest import checkpoint_dir, load_golden, relative_diff, requires_checkpoint

HIDDEN = 2048
KERNEL = 3
FIXTURE = Path(__file__).parent / "fixtures" / "lfm2_moe_forward.safetensors"
REPO = "LiquidAI/LFM2.5-8B-A1B"

DTYPES = [mx.float32, mx.bfloat16]

MIX = ConvMix(hidden=HIDDEN, kernel=KERNEL)


def ops_step(
    x: mx.array, weights: mx.array, taps: mx.array, window: mx.array
) -> tuple[mx.array, mx.array]:
    """`LFM2Conv.__call__` at T=1, up to (not including) out_proj — the reference path."""
    dtype = x.dtype
    b, c, v = mx.split((x[None, None, :] @ weights.T).astype(dtype), 3, axis=-1)
    bx = b * v
    padded = mx.concatenate([window[None], bx], axis=1)
    lifted = padded.astype(mx.float32)
    tap = taps.reshape(HIDDEN, KERNEL)
    conv = lifted[:, :1, :] * tap[:, 0]
    for j in range(1, KERNEL):
        conv = conv + lifted[:, j : j + 1, :] * tap[:, j]
    return (c * conv.astype(dtype))[0, 0], padded[0, 1:, :]


def onehot_case(dtype: mx.Dtype, seed: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """A case whose projections are exact in any reduction order: one nonzero per row.

    Magnitudes match the checkpoint's (hidden states ~N(0, 1) after the norm, taps
    inside [-1, 1]), so the products land where a T ulp actually separates values.
    """
    rng = np.random.default_rng(seed)
    dense = np.zeros((KERNEL * HIDDEN, HIDDEN), np.float32)
    for block in range(KERNEL):
        cols = rng.permutation(HIDDEN)
        dense[block * HIDDEN + np.arange(HIDDEN), cols] = rng.normal(size=HIDDEN)
    weights = mx.array(dense).astype(dtype)
    x = mx.array(rng.normal(size=HIDDEN).astype(np.float32)).astype(dtype)
    taps = mx.array(rng.normal(scale=0.5, size=KERNEL * HIDDEN).astype(np.float32)).astype(dtype)
    window = mx.array(rng.normal(size=(2, HIDDEN)).astype(np.float32)).astype(dtype)
    return x, weights, taps, window


@pytest.fixture(scope="module")
def checkpoint() -> tuple[mx.array, mx.array]:
    weights = mx.load(str(checkpoint_dir(REPO) / "model.safetensors"))
    assert isinstance(weights, dict)
    in_proj = weights["model.layers.0.conv.in_proj.weight"]
    taps = weights["model.layers.0.conv.conv.weight"].reshape(-1)
    mx.eval(in_proj, taps)
    return in_proj, taps


@pytest.fixture(scope="module")
def hidden_state() -> mx.array:
    """The fp32 input of layer 0's conv from the transformers fixture."""
    return load_golden(FIXTURE)["b0_ln_1"][0, -1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_against_ops_path(dtype: mx.Dtype) -> None:
    x, weights, taps, window = onehot_case(dtype, seed=0)
    gated, slid = MIX(x, weights, taps, window)
    reference, reference_window = ops_step(x, weights, taps, window)
    assert relative_diff(gated, reference) == 0
    assert mx.array_equal(slid, reference_window).item()


@pytest.mark.parametrize("dtype", DTYPES)
def test_window_slides_over_steps(dtype: mx.Dtype) -> None:
    """Three steps: the window the kernel emits must keep driving the op path exactly."""
    _, weights, taps, window = onehot_case(dtype, seed=1)
    ours, theirs = window, window
    for step in range(3):
        x = onehot_case(dtype, seed=10 + step)[0]
        gated, ours = MIX(x, weights, taps, ours)
        reference, theirs = ops_step(x, weights, taps, theirs)
        assert relative_diff(gated, reference) == 0
        assert mx.array_equal(ours, theirs).item()


@requires_checkpoint(REPO)
@pytest.mark.parametrize("dtype", DTYPES)
def test_checkpoint_step(
    checkpoint: tuple[mx.array, mx.array], hidden_state: mx.array, dtype: mx.Dtype
) -> None:
    """Real layer-0 weights and a real hidden state; only the gemv's order differs.

    The bound is the projection's own measured floor against fp64 — the fused dot
    reorders the same reduction, so it cannot be asked for better.
    """
    in_proj, taps = checkpoint
    w, t, x = in_proj.astype(dtype), taps.astype(dtype), hidden_state.astype(dtype)
    window = mx.concatenate([hidden_state[None] * 0.3, hidden_state[None] * -0.2]).astype(dtype)

    gated, slid = MIX(x, w, t, window)
    reference, reference_window = ops_step(x, w, t, window)

    exact = np.asarray(hidden_state, np.float64) @ np.asarray(
        in_proj.astype(mx.float32), np.float64
    ).T
    projected = np.asarray((x[None] @ w.T).astype(dtype).astype(mx.float32), np.float64)
    floor = float(np.max(np.abs(projected - exact)) / np.max(np.abs(exact)))

    if dtype == mx.bfloat16:
        assert relative_diff(gated, reference) == 0
        assert mx.array_equal(slid, reference_window).item()
    else:
        assert relative_diff(gated, reference) <= floor
        assert relative_diff(slid, reference_window) <= floor


MUTATIONS = {
    "taps_swapped": (
        "float conv = (float)TAPS[c * 3 + 0] * (float)WIN[c];",
        "float conv = (float)TAPS[c * 3 + 1] * (float)WIN[c];",
    ),
    "bx_not_rounded": (
        "float bx = (float)(T)((float)(T)B[r] * (float)(T)Xp[r]);",
        "float bx = (float)(T)B[r] * (float)(T)Xp[r];",
    ),
    "conv_not_rounded_before_gate": (
        "GATED[c] = (T)((float)(T)C[r] * (float)(T)conv);",
        "GATED[c] = (T)((float)(T)C[r] * conv);",
    ),
    "window_not_slid": ("WOUT[c] = WIN[h + c];", "WOUT[c] = WIN[c];"),
    "conv_accumulated_in_T": (
        "conv = conv + (float)TAPS[c * 3 + 1] * (float)WIN[h + c];",
        "conv = (float)(T)(conv + (float)TAPS[c * 3 + 1] * (float)WIN[h + c]);",
    ),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_mutation_is_caught(mutation: str) -> None:
    """Each documented break has to move the numbers the exact tests pin."""
    old, new = MUTATIONS[mutation]
    assert old in cm._SOURCE
    broken = metal_kernel(
        name=f"conv_mix_{mutation}",
        input_names=["X", "W", "TAPS", "WIN", "HD"],
        output_names=["GATED", "WOUT"],
        source=cm._SOURCE.replace(old, new),
    )
    for dtype in DTYPES:
        x, weights, taps, window = onehot_case(dtype, seed=0)
        gated, slid = broken(
            inputs=[x, weights, taps, window, mx.array(HIDDEN, mx.int32)],
            template=[("T", dtype)],
            grid=(32, HIDDEN // 2, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(HIDDEN,), (2, HIDDEN)],
            output_dtypes=[dtype, dtype],
        )
        reference, reference_window = ops_step(x, weights, taps, window)
        if relative_diff(gated, reference) > 0 or not mx.array_equal(slid, reference_window).item():
            return
    pytest.fail(f"{mutation} left the outputs unchanged in every dtype")


@pytest.mark.parametrize("dtype", DTYPES)
def test_default_matches_ops_path(dtype: mx.Dtype) -> None:
    """The default strategy is the op chain: same boundaries, same numbers."""
    x, weights, taps, window = onehot_case(dtype, seed=2)
    default = DefaultConvMix.build(hidden=HIDDEN, kernel=KERNEL, proj_bias=None, conv_bias=None)
    gated, slid = default(x, weights, taps, window)
    reference, reference_window = ops_step(x, weights, taps, window)
    assert relative_diff(gated, reference) == 0
    assert mx.array_equal(slid, reference_window).item()


def test_delegator_resolution() -> None:
    """The fused strategy takes the declaration it serves; everything else falls back."""
    assert isinstance(MIX.strategy, FusedConvMix)
    biased = ConvMix(hidden=HIDDEN, kernel=KERNEL, conv_bias=mx.zeros((HIDDEN,)))
    assert isinstance(biased.strategy, DefaultConvMix)
    assert isinstance(ConvMix(hidden=HIDDEN, kernel=4).strategy, DefaultConvMix)
    assert isinstance(ConvMix(hidden=HIDDEN + 2, kernel=KERNEL).strategy, DefaultConvMix)


def test_applies_predicate() -> None:
    assert cm.applies(HIDDEN, KERNEL, has_bias=False)
    assert not cm.applies(HIDDEN, 4, has_bias=False)
    assert not cm.applies(HIDDEN, KERNEL, has_bias=True)
    assert not cm.applies(HIDDEN + 2, KERNEL, has_bias=False)
