"""Parity and mutation gates for the fused gated delta rule.

The reference is the model's own ops implementation (`models.qwen3_5.delta_rule`) —
the same recurrence transformers' `torch_recurrent_gated_delta_rule` runs. Shapes are
the checkpoints': Qwen3.5-0.8B (Hk = Hv = 16, Dk = Dv = 128, kv-ratio 1) and
Qwen3.6-27B (Hk = 16, Hv = 48, ratio 3 — the only one that exercises the in-kernel
key-head broadcast). Magnitudes are checkpoint-like: q/k come out of `l2norm` (unit
rows), v out of conv+silu, beta out of a sigmoid and the decay out of
`exp(-exp(A_log)·softplus(dt))`, so a lane's dot lands where a bf16 ulp actually
separates something.
"""

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from conftest import relative_diff

from mlx_omnia.engine.core.kernels.gated_delta import (
    DefaultGatedDelta,
    FusedGatedDelta,
    GatedDelta,
    delta_rule,
    gated_delta,
    gated_delta_applies,
)
from mlx_omnia.engine.core.kernels.gated_delta.fused import _KERNEL, _SOURCE
from mlx_omnia.engine.core.layers import l2norm
from mlx_omnia.engine.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from mlx_omnia.engine.core.mxcompat import MetalKernel

SMALL = (16, 16, 128, 128)  # 0.8B: Hk, Hv, Dk, Dv
BROADCAST = (16, 48, 128, 128)  # 27B


class Inputs:
    """One DeltaNet step's inputs in both conventions: the ops path takes the log decay,
    an unscaled q and a `[B, Hv, Dk, Dv]` state with the key heads already repeated; the
    kernel takes the decay past the exp, q pre-scaled, an unrepeated q/k and the state
    transposed to `[B, Hv, Dv, Dk]`."""

    def __init__(self, length: int, shape: tuple[int, int, int, int], dtype: mx.Dtype) -> None:
        hk, hv, dk, dv = shape
        rng = np.random.default_rng(0xDE17A)

        def normal(*dims: int) -> mx.array:
            return mx.array(rng.standard_normal(dims), dtype=mx.float32)

        self.q = l2norm(normal(1, length, hk, dk)).astype(dtype)
        self.k = l2norm(normal(1, length, hk, dk)).astype(dtype)
        raw = normal(1, length, hv, dv) * 2.0
        self.v = (raw * mx.sigmoid(raw)).astype(dtype)
        self.beta = mx.sigmoid(normal(1, length, hv)).astype(dtype)
        self.g = -mx.exp(normal(hv) * 0.25) * nn.softplus(normal(1, length, hv) - 2.0)
        self.state = normal(1, hv, dk, dv) * 0.1
        self.ratio = hv // hk
        self.scale = 1 / math.sqrt(dk)

    def reference(self) -> tuple[mx.array, mx.array]:
        q = mx.repeat(self.q, self.ratio, axis=2)
        k = mx.repeat(self.k, self.ratio, axis=2)
        return delta_rule(q, k, self.v, self.g, self.beta, self.state)

    def kernel_args(self) -> tuple[mx.array, ...]:
        return (
            self.q.astype(mx.float32) * self.scale,
            self.k,
            self.v,
            mx.exp(self.g),
            self.beta,
            self.state.transpose(0, 1, 3, 2),
        )


def dispatch(kernel: "MetalKernel", args: tuple[mx.array, ...]) -> tuple[mx.array, mx.array]:
    """The host side of `gated_delta`, kept separate so a mutated source runs it too."""
    q, k, v, g, beta, state = args
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    out = kernel(
        inputs=[q, k, v, g, beta, state, mx.array(length, dtype=mx.int32)],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", key_heads),
            ("Hv", value_heads),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, length, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
    return out[0], out[1]


def run(args: tuple[mx.array, ...]) -> tuple[mx.array, mx.array]:
    y, state = dispatch(_KERNEL, args)
    return y, state.transpose(0, 1, 3, 2)


def test_applies_predicate() -> None:
    hk, hv, dk, dv = BROADCAST
    assert gated_delta_applies(dk, hk, hv, dv)
    assert not gated_delta_applies(100, hk, hv, dv)
    assert not gated_delta_applies(dk, 7, hv, dv)
    assert not gated_delta_applies(dk, hk, hv, 66)


@pytest.mark.parametrize("length", [1, 7, 64])
@pytest.mark.parametrize("shape", [SMALL, BROADCAST], ids=["ratio1", "ratio3"])
def test_fp32_parity(length: int, shape: tuple[int, int, int, int]) -> None:
    """fp32 template against the ops chain. Only the reduction order differs (per-lane
    accumulation + simd_sum vs mlx's tree sum), so the house 1e-5 holds."""
    inputs = Inputs(length, shape, mx.float32)
    y, state = run(inputs.kernel_args())
    ref_y, ref_state = inputs.reference()
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state, ref_state) < 1e-5


def test_bf16_within_measured_floor() -> None:
    """bf16 against the fp32 ops reference, held to 3x what the bf16 ops chain itself
    costs against that same reference — the floor is measured here, never guessed."""
    fp32 = Inputs(64, BROADCAST, mx.float32)
    bf16 = Inputs(64, BROADCAST, mx.bfloat16)
    ref_y, ref_state = fp32.reference()
    ops_y, ops_state = bf16.reference()
    floor_y = relative_diff(ops_y, ref_y)
    floor_state = relative_diff(ops_state, ref_state)
    assert floor_y > 0.0

    y, state = run(bf16.kernel_args())
    assert relative_diff(y, ref_y) < 3 * floor_y
    assert relative_diff(state, ref_state) < 3 * floor_state


def test_step_by_step_matches_one_dispatch() -> None:
    """The recurrence is carried in registers across T; feeding the tokens one at a time
    does the identical arithmetic in the identical order, so this is bit-exact."""
    inputs = Inputs(16, BROADCAST, mx.float32)
    q, k, v, g, beta, state = inputs.kernel_args()
    steps: list[mx.array] = []
    for t in range(16):
        y, state = dispatch(
            _KERNEL,
            (q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
             g[:, t : t + 1], beta[:, t : t + 1], state),
        )
        steps.append(y)
    whole_y, whole_state = dispatch(_KERNEL, inputs.kernel_args())
    assert mx.array_equal(mx.concatenate(steps, axis=1), whole_y).item()
    assert mx.array_equal(state, whole_state).item()


def test_public_entry_point_matches() -> None:
    inputs = Inputs(8, BROADCAST, mx.float32)
    y, state = gated_delta(*inputs.kernel_args())
    ref_y, ref_state = inputs.reference()
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state.transpose(0, 1, 3, 2), ref_state) < 1e-5


def test_default_strategy_matches_reference() -> None:
    """The ops strategy takes the kernel's convention (decay past the exp, q pre-scaled,
    q/k unrepeated, `[B, Hv, Dv, Dk]` state) and must land on `delta_rule`'s output —
    the adaptation is transposes and a repeat, no arithmetic."""
    inputs = Inputs(8, BROADCAST, mx.float32)
    hk, hv, dk, dv = BROADCAST
    strategy = DefaultGatedDelta.build(
        key_dim=dk, key_heads=hk, value_heads=hv, value_dim=dv, enabled=False
    )
    y, state = strategy(*inputs.kernel_args())
    ref_y, ref_state = inputs.reference()
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state.transpose(0, 1, 3, 2), ref_state) < 1e-5


def test_default_strategy_per_channel_decay() -> None:
    """The per-channel decay branch, against the kernel variant that serves it."""
    inputs = Inputs(8, SMALL, mx.float32)
    hk, hv, dk, dv = SMALL
    rng = np.random.default_rng(0xDECA7)
    g = mx.array(np.exp(-np.abs(rng.standard_normal((1, 8, hv, dk))) * 0.1), mx.float32)
    q, k, v, _g, beta, state = inputs.kernel_args()
    args = (q, k, v, g, beta, state)
    fused = FusedGatedDelta.build(
        key_dim=dk, key_heads=hk, value_heads=hv, value_dim=dv, enabled=True
    )
    assert fused is not None
    ops = DefaultGatedDelta.build(
        key_dim=dk, key_heads=hk, value_heads=hv, value_dim=dv, enabled=True
    )
    y, state_out = fused(*args)
    ref_y, ref_state = ops(*args)
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state_out, ref_state) < 1e-5


@pytest.mark.parametrize("enabled", [True, False])
def test_delegator_resolves_and_matches(enabled: bool) -> None:
    """Total either way: the kernel when its tiling admits the shapes and the switch is
    on, the ops recurrence otherwise, same call and same result."""
    hk, hv, dk, dv = BROADCAST
    facade = GatedDelta(
        key_dim=dk, key_heads=hk, value_heads=hv, value_dim=dv, enabled=enabled
    )
    expected = FusedGatedDelta if enabled else DefaultGatedDelta
    assert isinstance(facade.strategy, expected)

    inputs = Inputs(8, BROADCAST, mx.float32)
    y, state = facade(*inputs.kernel_args())
    ref_y, ref_state = inputs.reference()
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state.transpose(0, 1, 3, 2), ref_state) < 1e-5


def test_delegator_falls_back_on_unfit_shapes() -> None:
    assert isinstance(
        GatedDelta(key_dim=100, key_heads=4, value_heads=4, value_dim=100).strategy,
        DefaultGatedDelta,
    )


MUTATIONS = {
    "broadcast_modulo": (
        "auto hk_idx = hv_idx / (Hv / Hk);",
        "auto hk_idx = hv_idx % Hk;",
    ),
    "decay_after_read": (
        "state[i] = state[i] * g_[hv_idx];\n        kv_mem += state[i] * k_[s_idx];",
        "kv_mem += state[i] * k_[s_idx];\n        state[i] = state[i] * g_[hv_idx];",
    ),
    "read_before_write": (
        "state[i] = state[i] + k_[s_idx] * delta;\n        out += state[i] * q_[s_idx];",
        "out += state[i] * q_[s_idx];\n        state[i] = state[i] + k_[s_idx] * delta;",
    ),
    "beta_dropped": ("* beta_[hv_idx];", "* 1.0f;"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_breaks_parity(name: str) -> None:
    """Break the kernel, confirm the failure. The ratio-3 shape is what makes the
    broadcast mutation visible at all — on the 0.8B (ratio 1) it is a no-op."""
    old, new = MUTATIONS[name]
    assert old in _SOURCE
    mutated = metal_kernel(
        name=f"gated_delta_mut_{name}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_SOURCE.replace(old, new),
    )
    inputs = Inputs(8, BROADCAST, mx.float32)
    y, _state = dispatch(mutated, inputs.kernel_args())
    ref_y, _ref_state = inputs.reference()
    assert relative_diff(y, ref_y) > 1e-3
