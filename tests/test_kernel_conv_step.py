"""The T=1 conv step's fused kernel against the op chain, bit-exact.

The op chain rounds at every boundary — each tap product, each accumulation, the bias,
the sigmoid, the final product — and the kernel claims to reproduce every one, so the
comparison is equality, not a tolerance, in both fp32 and bf16.
"""

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.kernels.conv_step import DefaultConvStep, FusedConvStep

CONV_DIM = 256
KERNEL = 4


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fused_matches_ops(dtype: mx.Dtype, with_bias: bool) -> None:
    mx.random.seed(7)
    taps = mx.random.normal((CONV_DIM, KERNEL)).astype(dtype)
    bias = mx.random.normal((CONV_DIM,)).astype(dtype) if with_bias else None
    x = mx.random.normal((CONV_DIM,)).astype(dtype)
    window = mx.random.normal((KERNEL - 1, CONV_DIM)).astype(dtype)

    fused = FusedConvStep.build(taps=taps, bias=bias, conv_dim=CONV_DIM, kernel=KERNEL)
    assert fused is not None
    reference = DefaultConvStep.build(taps=taps, bias=bias, conv_dim=CONV_DIM, kernel=KERNEL)
    assert reference is not None

    out, slid = fused(x, window)
    wanted, wanted_window = reference(x, window)
    assert mx.array_equal(out, wanted).item()
    assert mx.array_equal(slid, wanted_window).item()
