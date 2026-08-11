"""The MXFP4 routed expert MLP kernels against gpt-oss's `gather_qmm(mode="mxfp4")` path.

The replica is the **complete** T=1 step at gpt-oss-20b's shapes (32 experts, hidden
2880, inner 2880, top-4, swiglu limit 7): both dispatches chained, the per-row projection
biases, the routing weights, the expert sum and the residual — the same ops
`GPTOSSExperts.__call__` plus `GPTOSSMLP.__call__` run. Magnitudes are checkpoint-like
(e8m0 exponents around 2^-6, unit-variance activations) so the dots land where a bf16 ulp
actually separates something.
"""

import mlx.core as mx
import numpy as np
import pytest
from conftest import relative_diff

from mlx_omnia.core.kernels.down_combine.mxfp4 import Mxfp4DownCombine
from mlx_omnia.core.kernels.gate_up.mxfp4 import Mxfp4GateUp
from mlx_omnia.core.kernels.gate_up.mxfp4 import applies as mxfp4_moe_applies

EXPERTS, HIDDEN, INNER, TOPK, LIMIT = 32, 2880, 2880, 4, 7.0
GROUP = 32


class Stack:
    def __init__(self, weight: mx.array, scales: mx.array, bias: mx.array) -> None:
        self.weight, self.scales, self.bias = weight, scales, bias


def _pack(rng: np.random.Generator, rows: int, kdim: int) -> Stack:
    codes = rng.integers(0, 16, size=(EXPERTS, rows, kdim), dtype=np.uint32)
    words = codes.reshape(EXPERTS, rows, kdim // 8, 8)
    packed = np.zeros((EXPERTS, rows, kdim // 8), dtype=np.uint32)
    for n in range(8):
        packed |= words[..., n] << np.uint32(4 * n)
    # e8m0 exponents around 127-6: the magnitude an MXFP4 checkpoint actually carries.
    exponents = rng.integers(118, 124, size=(EXPERTS, rows, kdim // GROUP)).astype(np.uint8)
    bias = rng.standard_normal((EXPERTS, rows)).astype(np.float32) * 0.05
    return Stack(mx.array(packed), mx.array(exponents), mx.array(bias))


@pytest.fixture(scope="module")
def step() -> tuple[Stack, Stack, mx.array, mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(11)
    gate_up = _pack(rng, 2 * INNER, HIDDEN)
    down = _pack(rng, HIDDEN, INNER)
    x = mx.array(rng.standard_normal(HIDDEN).astype(np.float32))
    residual = mx.array(rng.standard_normal(HIDDEN).astype(np.float32))
    indices = mx.array(rng.choice(EXPERTS, size=TOPK, replace=False).astype(np.uint32))
    routing = mx.array(np.array([0.42, 0.31, 0.19, 0.08], dtype=np.float32))
    mx.eval(gate_up.weight, gate_up.scales, gate_up.bias, down.weight, down.scales, down.bias)
    return gate_up, down, x, residual, indices, routing


def _gather(x: mx.array, stack: Stack, indices: mx.array, dtype: mx.Dtype) -> mx.array:
    projected = mx.gather_qmm(
        x, stack.weight, stack.scales.astype(mx.uint8), None, rhs_indices=indices,
        transpose=True, group_size=GROUP, bits=4, mode="mxfp4",
    )
    return projected + mx.expand_dims(stack.bias.astype(dtype)[indices], axis=-2)


def reference_step(
    gate_up: Stack, down: Stack, x: mx.array, residual: mx.array,
    indices: mx.array, routing: mx.array, dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    """`GPTOSSExperts.__call__` + the routing-weighted sum + residual, ops only."""
    fused = _gather(x.astype(dtype)[None, None, None], gate_up, indices[None], dtype)
    pairs = fused.reshape(1, TOPK, INNER, 2)
    gate = mx.clip(pairs[..., 0], None, LIMIT)
    up = mx.clip(pairs[..., 1], -LIMIT, LIMIT)
    act = gate * mx.sigmoid(1.702 * gate) * (up + 1)
    projected = _gather(act[:, :, None], down, indices[None], dtype).squeeze(2)
    combined = (projected * routing.astype(dtype)[None, :, None]).sum(axis=1)
    return act.reshape(TOPK, INNER), combined.reshape(-1) + residual.astype(dtype)


def kernel_step(
    gate_up: Stack, down: Stack, x: mx.array, residual: mx.array,
    indices: mx.array, routing: mx.array, dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    act = Mxfp4GateUp(gate_up.weight, gate_up.scales, gate_up.bias.astype(dtype), LIMIT)(
        x.astype(dtype), indices
    )
    out = Mxfp4DownCombine(down.weight, down.scales, down.bias.astype(dtype))(
        act, indices, routing.astype(dtype), residual.astype(dtype)
    )
    return act, out


def test_applies_predicate() -> None:
    assert mxfp4_moe_applies(HIDDEN, INNER)
    assert not mxfp4_moe_applies(2880, 100)
    assert not mxfp4_moe_applies(24, 2880)


def test_matches_op_chain_fp32(
    step: tuple[Stack, Stack, mx.array, mx.array, mx.array, mx.array],
) -> None:
    """The fp32 template of the real kernels against the model's gather_qmm chain."""
    act, out = kernel_step(*step, mx.float32)
    ref_act, reference = reference_step(*step, mx.float32)
    assert relative_diff(act, ref_act) < 1e-5
    assert relative_diff(out, reference) < 1e-5


def test_matches_op_chain_bfloat16(
    step: tuple[Stack, Stack, mx.array, mx.array, mx.array, mx.array],
) -> None:
    """bf16 against bf16: the kernel keeps the whole contraction in one f32 accumulator
    where gather_qmm reduces per tile, so the bound is the measured reassociation floor
    (6.7e-3 on the activations, 1.0e-2 on the block output), not zero. Both are within
    two units of 2^-7, the resolution of max|a-b|/max|b| in bf16."""
    act, out = kernel_step(*step, mx.bfloat16)
    ref_act, reference = reference_step(*step, mx.bfloat16)
    assert relative_diff(act, ref_act) < 7.8e-3
    assert relative_diff(out, reference) < 1.6e-2


@pytest.mark.parametrize("mutation", ["sign", "scale", "indices", "gate_up_swap", "bias"])
def test_mutations_break_parity(
    step: tuple[Stack, Stack, mx.array, mx.array, mx.array, mx.array], mutation: str
) -> None:
    """Each mutation feeds the kernel a stack the reference does not see; the fp32 gap
    must leave the 1e-5 parity band. Covers the four things the port can get wrong: the
    sign nibble of the e2m1 lookup, the e8m0 exponent, the expert indirection, and the
    gate/up row interleave (plus the projection bias mlx's mxfp4 mode has no slot for)."""
    gate_up, down, x, residual, indices, routing = step
    _, reference = reference_step(*step, mx.float32)
    broken = Stack(gate_up.weight, gate_up.scales, gate_up.bias)
    idx = indices
    if mutation == "sign":
        broken = Stack(gate_up.weight ^ mx.array(0x88888888, dtype=mx.uint32),
                       gate_up.scales, gate_up.bias)
    elif mutation == "scale":
        broken = Stack(gate_up.weight, gate_up.scales + mx.array(1, dtype=mx.uint8),
                       gate_up.bias)
    elif mutation == "indices":
        idx = indices[mx.array(np.array([1, 0, 3, 2], dtype=np.uint32))]
    elif mutation == "gate_up_swap":
        order = mx.array((np.arange(2 * INNER) ^ 1).astype(np.uint32))
        broken = Stack(gate_up.weight[:, order], gate_up.scales[:, order],
                       gate_up.bias[:, order])
    else:
        broken = Stack(gate_up.weight, gate_up.scales, gate_up.bias + 1.0)

    act = Mxfp4GateUp(broken.weight, broken.scales, broken.bias, LIMIT)(x, idx)
    out = Mxfp4DownCombine(down.weight, down.scales, down.bias)(act, idx, routing, residual)
    assert relative_diff(out, reference) > 1e-5
