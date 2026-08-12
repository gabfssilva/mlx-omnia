"""The fused Mamba2 T=1 middle against the op chain, at Nemotron-3.5's dimensions.

Everything but the norm's reduction is claimed exact, and the norm's fp32
reduction-order noise is orders below the bound here: the comparison-side arithmetic
runs in fp32 and the bound is 1e-5 relative, the fp32 stepwise floor.
"""

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.kernels.mamba_step import DefaultMambaStep, FusedMambaStep

INNER = 4096
HEADS = 64
HEAD_DIM = 64
GROUPS = 8
STATE = 128
KERNEL = 4
CONV = INNER + 2 * GROUPS * STATE
EPS = 1e-5
LIMITS = (0.0, float("inf"))


def _relative(a: mx.array, b: mx.array) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    diff = (mx.abs(a32 - b32).max() / mx.abs(b32).max()).item()
    assert isinstance(diff, float)
    return diff


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_matches_ops(dtype: mx.Dtype, with_bias: bool) -> None:
    mx.random.seed(13)
    taps = (mx.random.normal((CONV, KERNEL)) * 0.3).astype(dtype)
    conv_bias = (mx.random.normal((CONV,)) * 0.3).astype(dtype) if with_bias else None
    A_log = mx.random.normal((HEADS,)) * 0.3
    D = mx.random.normal((HEADS,)) * 0.3
    dt_bias = mx.random.normal((HEADS,)) * 0.3
    norm_weight = (mx.ones((INNER,)) + mx.random.normal((INNER,)) * 0.2).astype(dtype)
    leaves = dict(
        taps=taps, conv_bias=conv_bias, A_log=A_log, D=D, dt_bias=dt_bias,
        norm_weight=norm_weight, eps=EPS, inner=INNER, conv_dim=CONV, kernel=KERNEL,
        heads=HEADS, head_dim=HEAD_DIM, groups=GROUPS, state_size=STATE,
        time_step_limit=LIMITS,
    )
    fused = FusedMambaStep.build(**leaves)
    assert fused is not None
    reference = DefaultMambaStep.build(**leaves)
    assert reference is not None

    proj = (mx.random.normal((INNER + CONV + HEADS,)) * 0.5).astype(dtype)
    window = (mx.random.normal((KERNEL - 1, CONV)) * 0.5).astype(dtype)
    state = mx.random.normal((HEADS, HEAD_DIM, STATE)).astype(mx.float32)

    out, slid, new_state = fused(proj, window, state)
    wanted, wanted_window, wanted_state = reference(proj, window, state)
    assert mx.array_equal(slid, wanted_window).item()
    assert _relative(new_state, wanted_state) < 1e-5
    assert _relative(out, wanted) < 1e-5


def test_verify_matches_iterated_steps() -> None:
    """T=3 through the verify kernel equals three iterated T=1 steps, token by token,
    including every per-token state slot."""
    from mlx_omnia.engine.core.kernels.mamba_step.verify import VerifyMambaStep

    mx.random.seed(17)
    dtype = mx.bfloat16
    taps = (mx.random.normal((CONV, KERNEL)) * 0.3).astype(dtype)
    conv_bias = (mx.random.normal((CONV,)) * 0.3).astype(dtype)
    A_log = mx.random.normal((HEADS,)) * 0.3
    D = mx.random.normal((HEADS,)) * 0.3
    dt_bias = mx.random.normal((HEADS,)) * 0.3
    norm_weight = (mx.ones((INNER,)) + mx.random.normal((INNER,)) * 0.2).astype(dtype)
    leaves = dict(
        taps=taps, conv_bias=conv_bias, A_log=A_log, D=D, dt_bias=dt_bias,
        norm_weight=norm_weight, eps=EPS, inner=INNER, conv_dim=CONV, kernel=KERNEL,
        heads=HEADS, head_dim=HEAD_DIM, groups=GROUPS, state_size=STATE,
        time_step_limit=LIMITS,
    )
    fused = FusedMambaStep.build(**leaves)
    assert fused is not None
    verify = VerifyMambaStep.of(fused)
    reference = DefaultMambaStep.build(**leaves)
    assert reference is not None

    tokens = 3
    proj = (mx.random.normal((tokens, INNER + CONV + HEADS)) * 0.5).astype(dtype)
    window = (mx.random.normal((KERNEL - 1, CONV)) * 0.5).astype(dtype)
    state = mx.random.normal((HEADS, HEAD_DIM, STATE)).astype(mx.float32)

    out, slots = verify(proj, window, state)

    rolling_window, rolling_state = window, state
    for t in range(tokens):
        wanted, rolling_window, rolling_state = reference(
            proj[t], rolling_window, rolling_state
        )
        assert _relative(out[t], wanted) < 1e-5, t
        assert _relative(slots[t], rolling_state) < 1e-5, t
