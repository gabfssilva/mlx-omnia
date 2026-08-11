"""Parity and mutation gates for the SSD decode kernel.

The reference is the ops implementation (`core.kernels.ssm.ssm_step_ref`)
— the same recurrence transformers' naive `torch_forward` runs token-by-token.
Shapes are the Codestral-Mamba-7B's: 128 heads, 64 head_dim, 128 state, 8 groups
(heads_per_group = 16). Magnitudes are checkpoint-like: x out of conv+silu, B/C
out of the conv split, dt pre-softplus, A_log and D in fp32.
"""

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
import pytest
from conftest import relative_diff

from sideros.core.kernels.ssm import ssm_step_ref
from sideros.core.kernels.ssm.step import _KERNEL, _SOURCE, ssm_step, ssm_step_applies
from sideros.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from sideros.core.mxcompat import MetalKernel

HEADS, HEAD_DIM, STATE, GROUPS = 128, 64, 128, 8
LIMIT = (0.0, float("inf"))

type SSMArgs = tuple[
    mx.array, mx.array, mx.array, mx.array, mx.array,
    mx.array, mx.array, mx.array, tuple[float, float],
]


class SSMInputs:
    """One SSD step's inputs in both conventions: the ops path takes the raw dt
    (pre-softplus), the full `[B, T, H, Dh]` x and `[B, T, G, Ds]` B/C; the kernel
    takes the same shapes (compute_dt runs inside `ssm_step`)."""

    def __init__(self, length: int, dtype: mx.Dtype) -> None:
        rng = np.random.default_rng(0x5D)

        def normal(*dims: int) -> mx.array:
            return mx.array(rng.standard_normal(dims), dtype=mx.float32)

        self.x = (normal(1, length, HEADS, HEAD_DIM) * 0.5).astype(dtype)
        self.A_log = normal(HEADS) * 0.1
        self.dt_bias = normal(HEADS) * 0.1
        self.D = mx.ones(HEADS, dtype=mx.float32)
        self.B = normal(1, length, GROUPS, STATE) * 0.3
        self.C = normal(1, length, GROUPS, STATE) * 0.3
        self.dt = normal(1, length, HEADS).astype(dtype)
        self.state = normal(1, HEADS, HEAD_DIM, STATE) * 0.1

    def reference(self) -> tuple[mx.array, mx.array]:
        return ssm_step_ref(
            self.x, self.A_log, self.B, self.C, self.D,
            self.dt, self.dt_bias, self.state, LIMIT,
        )

    def kernel_args(self) -> SSMArgs:
        return (
            self.x, self.A_log, self.B, self.C, self.D,
            self.dt, self.dt_bias, self.state, LIMIT,
        )


def dispatch(
    kernel: "MetalKernel", args: SSMArgs
) -> tuple[mx.array, mx.array]:
    """The host side of `ssm_step`, kept separate so a mutated source runs it too."""
    x, A_log, B, C, D, dt, dt_bias, state, limit = args
    batch = x.shape[0]
    num_heads = x.shape[2]
    head_dim = x.shape[3]
    n_groups = B.shape[2]
    state_size = B.shape[3]
    from sideros.core.kernels.ssm.step import _compute_dt
    dt_processed = _compute_dt(dt, dt_bias, limit)
    heads_per_group = num_heads // n_groups
    out, state_out = kernel(
        inputs=[x, A_log, B, C, D, dt_processed, state],
        template=[
            ("T", x.dtype),
            ("U", state.dtype),
            ("Dh", head_dim),
            ("Ds", state_size),
            ("H", num_heads),
            ("G", heads_per_group),
        ],
        grid=(32, head_dim, batch * num_heads),
        threadgroup=(32, 8, 1),
        output_shapes=[(batch, 1, num_heads, head_dim), state.shape],
        output_dtypes=[x.dtype, state.dtype],
    )
    return out, state_out


def run(args: SSMArgs) -> tuple[mx.array, mx.array]:
    return dispatch(_KERNEL, args)


def test_applies_predicate() -> None:
    assert ssm_step_applies(STATE, HEADS, GROUPS)
    assert not ssm_step_applies(100, HEADS, GROUPS)
    assert not ssm_step_applies(STATE, 7, GROUPS)


@pytest.mark.parametrize("length", [1, 7])
def test_fp32_parity(length: int) -> None:
    """fp32 template against the ops chain. Only the reduction order differs
    (per-lane accumulation + simd_sum vs mlx's tree sum), so the house 1e-5 holds."""
    inputs = SSMInputs(length, mx.float32)
    y, state = run(inputs.kernel_args())
    ref_y, ref_state = inputs.reference()
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state, ref_state) < 1e-5


def test_bf16_within_measured_floor() -> None:
    """bf16 against the fp32 ops reference, held to 3x what the bf16 ops chain
    itself costs against that same reference — the floor is measured here."""
    fp32 = SSMInputs(7, mx.float32)
    bf16 = SSMInputs(7, mx.bfloat16)
    ref_y, ref_state = fp32.reference()
    ops_y, ops_state = bf16.reference()
    floor_y = relative_diff(ops_y, ref_y)
    floor_state = relative_diff(ops_state, ref_state)
    assert floor_y > 0.0

    y, state = run(bf16.kernel_args())
    assert relative_diff(y, ref_y) < 3 * floor_y
    assert relative_diff(state, ref_state) < 3 * floor_state


def test_step_by_step_matches_one_dispatch() -> None:
    """Feeding the tokens one at a time does the identical arithmetic in the
    identical order, so this is bit-exact."""
    inputs = SSMInputs(16, mx.float32)
    x, A_log, B, C, D, dt, dt_bias, state, limit = inputs.kernel_args()
    steps: list[mx.array] = []
    for t in range(16):
        y, state = dispatch(
            _KERNEL,
            (x[:, t : t + 1], A_log, B[:, t : t + 1], C[:, t : t + 1],
             D, dt[:, t : t + 1], dt_bias, state, limit),
        )
        steps.append(y)
    whole_y, whole_state = run(inputs.kernel_args())
    assert mx.array_equal(mx.concatenate(steps, axis=1), whole_y).item()
    assert mx.array_equal(state, whole_state).item()


def test_public_entry_point_matches() -> None:
    inputs = SSMInputs(8, mx.float32)
    y, state = ssm_step(*inputs.kernel_args())
    ref_y, ref_state = inputs.reference()
    assert relative_diff(y, ref_y) < 1e-5
    assert relative_diff(state, ref_state) < 1e-5


MUTATIONS = {
    "group_broadcast": (
        "auto g_idx = n / G;",
        "auto g_idx = n % G;",
    ),
    "d_skip_dropped": (
        "out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);",
        "out[d_idx] = static_cast<T>(acc);",
    ),
    "decay_after_update": (
        "auto state = dA * i_state[idx] + dB_by_x;",
        "auto state = i_state[idx] + dB_by_x;",
    ),
    "b_dropped": (
        "auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);",
        "auto dB_by_x = x_ * dt_ * 1.0f;",
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_breaks_parity(name: str) -> None:
    old, new = MUTATIONS[name]
    assert old in _SOURCE
    mutated = metal_kernel(
        name=f"ssm_step_mut_{name}",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=_SOURCE.replace(old, new),
    )
    inputs = SSMInputs(8, mx.float32)
    y, _state = dispatch(mutated, inputs.kernel_args())
    ref_y, _ref_state = inputs.reference()
    assert relative_diff(y, ref_y) > 1e-3
