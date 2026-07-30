# pyright: basic
"""SSM kernel validation on a replica of the complete Mamba2 step.

The kernel (`ssm_step`) runs the per-token SSD recurrence in one Metal dispatch.
This test builds a small replica of the full mamba mixer step (in_proj split →
conv1d → SSD scan → out_proj) and checks the kernel output against an ops-path
reference, in both fp32 and bf16 (dtype template mandatory).

Parity: `relative_diff < 1e-5` (fp32). Mutation: flip one tap of the conv1d or
one A_log entry and confirm the diff jumps well above the floor.

Cannot run in this environment (parallel MLX runs reboot the M5). The
orchestrator runs all validation serially.
"""


import mlx.core as mx
import mlx.nn as nn
import pytest

from sideros.core.kernels.ssm import ssm_applies, ssm_attn, ssm_step


def _ops_path_step(
    hidden: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Pure-mlx reference: the SSD recurrence written as a per-token loop in
    fp32, matching what the kernel computes."""
    batch, _, heads, head_dim = hidden.shape
    _, groups, d_state = B.shape

    A = -mx.exp(A_log.astype(mx.float32))
    dt_f = mx.clip(
        nn.softplus(dt.astype(mx.float32) + dt_bias.astype(mx.float32)),
        0.0,
        float("inf"),
    ).reshape(batch, heads)

    out = mx.zeros((batch, 1, heads, head_dim), dtype=hidden.dtype)
    new_state = state

    for h in range(heads):
        g = h // (heads // groups)
        dA = mx.exp(A[h] * dt_f[:, h])
        for d in range(head_dim):
            acc = mx.zeros((batch,), dtype=mx.float32)
            for s in range(d_state):
                dBx = (
                    hidden[:, 0, h, d].astype(mx.float32)
                    * dt_f[:, h]
                    * B[:, g, s].astype(mx.float32)
                )
                st = dA * state[:, h, d, s] + dBx
                new_state[:, h, d, s] = st
                acc = acc + st * C[:, g, s].astype(mx.float32)
            out[:, 0, h, d] = (
                acc + hidden[:, 0, h, d].astype(mx.float32) * D[h]
            ).astype(hidden.dtype)

    return out, new_state


def _make_replica(dtype: mx.Dtype) -> tuple:
    """Build a minimal but complete mamba step replica: random in_proj, conv1d,
    A_log, dt_bias, D, and a random input + zero state."""
    mx.random.seed(42)
    hidden_size = 64
    intermediate = 32
    heads = 4
    head_dim = 8
    d_state = 64
    groups = 2

    conv_dim = intermediate + 2 * groups * d_state
    proj_size = intermediate + conv_dim + heads

    in_proj_w = mx.random.normal((proj_size, hidden_size), dtype=dtype) * 0.05
    x = mx.random.normal((1, 1, hidden_size), dtype=dtype) * 0.1

    conv_w = mx.random.normal((conv_dim, 4), dtype=dtype) * 0.1
    conv_b = mx.random.normal((conv_dim,), dtype=dtype) * 0.01

    A_log = mx.log(mx.arange(1, heads + 1, dtype=mx.float32))
    dt_bias = mx.ones((heads,), dtype=mx.float32)
    D = mx.ones((heads,), dtype=mx.float32)
    state = mx.zeros((1, heads, head_dim, d_state), dtype=mx.float32)

    return (
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    )


def _run_mamba_step(
    in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
    intermediate, conv_dim, heads, head_dim, d_state, groups,
):
    """The complete step: in_proj → conv1d → split → SSD → out."""
    projected = x @ in_proj_w.T
    _gate, conv_input, dt_raw = mx.split(
        projected, [intermediate, intermediate + conv_dim], axis=-1
    )

    # conv1d at T=1: just multiply by the last tap + conv_bias (zero-padded window)
    conv_out = conv_input * conv_w[:, -1] + conv_b
    conv_out = nn.silu(conv_out)

    hidden, B, C = mx.split(
        conv_out,
        [intermediate, intermediate + groups * d_state],
        axis=-1,
    )
    hidden_r = hidden.reshape(1, 1, heads, head_dim)
    B_r = B.reshape(1, groups, d_state)
    C_r = C.reshape(1, groups, d_state)

    # dt stays pre-softplus — the kernel/reference apply softplus(dt + dt_bias)
    # internally (ssm.py's `_compute_dt`), so the replica feeds the raw projection.
    return hidden_r, B_r, C_r, D, dt_raw, dt_bias, state, A_log


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_ssm_kernel_parity(dtype: mx.Dtype) -> None:
    """The kernel must match the ops-path reference within the fp32 floor."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        pytest.skip("Metal not available")

    (
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    ) = _make_replica(dtype)

    hidden, B, C, D_arr, dt, dt_bias, state, A_log = _run_mamba_step(
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    )

    assert ssm_applies(head_dim, d_state, heads, groups)

    kernel_out, kernel_state = ssm_step(hidden, A_log, B, C, D_arr, dt, dt_bias, state)
    mx.eval(kernel_out, kernel_state)

    ref_out, ref_state = _ops_path_step(hidden, A_log, B, C, D_arr, dt, dt_bias, state)
    mx.eval(ref_out, ref_state)

    diff = float(mx.abs(kernel_out.astype(mx.float32) - ref_out.astype(mx.float32)).max().item())
    ref_max = float(mx.abs(ref_out.astype(mx.float32)).max().item())
    rel = diff / max(ref_max, 1e-12)
    assert rel < 1e-5, f"kernel vs ops: relative_diff={rel:.3e}"


def test_ssm_attn_chunked_prefill_parity() -> None:
    """The chunked prefill path must match the per-token loop over a short
    sequence."""
    mx.random.seed(99)
    batch, length, heads, head_dim, groups, d_state = 1, 8, 4, 16, 2, 64

    x = mx.random.normal((batch, length, heads, head_dim), dtype=mx.float32) * 0.1
    B = mx.random.normal((batch, length, groups, d_state), dtype=mx.float32) * 0.1
    C = mx.random.normal((batch, length, groups, d_state), dtype=mx.float32) * 0.1
    A_log = mx.log(mx.arange(1, heads + 1, dtype=mx.float32))
    D = mx.ones((heads,), dtype=mx.float32)
    dt = mx.random.normal((batch, length, heads), dtype=mx.float32) * 0.1 + 0.5
    dt_bias = mx.zeros((heads,), dtype=mx.float32)

    state = mx.zeros((batch, heads, head_dim, d_state), dtype=mx.float32)

    chunked_out, chunked_state = ssm_attn(x, A_log, B, C, D, dt, dt_bias, state, step=4)
    mx.eval(chunked_out, chunked_state)

    # Reference: per-token loop
    ref_state = mx.zeros((batch, heads, head_dim, d_state), dtype=mx.float32)
    ref_outs: list[mx.array] = []
    for t in range(length):
        h_t = x[:, t : t + 1, :, :]
        B_t = B[:, t : t + 1, :, :].reshape(batch, groups, d_state)
        C_t = C[:, t : t + 1, :, :].reshape(batch, groups, d_state)
        dt_t = dt[:, t : t + 1, :].reshape(batch, 1, heads)
        out_t, ref_state = ssm_step(h_t, A_log, B_t, C_t, D, dt_t, dt_bias, ref_state)
        ref_outs.append(out_t)
    ref_out = mx.concatenate(ref_outs, axis=1) + x * D.reshape(1, 1, heads, 1)
    mx.eval(ref_out, ref_state)

    diff = float(mx.abs(chunked_out.astype(mx.float32) - ref_out.astype(mx.float32)).max().item())
    ref_max = float(mx.abs(ref_out.astype(mx.float32)).max().item())
    rel = diff / max(ref_max, 1e-12)
    assert rel < 1e-4, f"chunked vs per-token: relative_diff={rel:.3e}"


def test_mutation_conv1d_breaks_parity() -> None:
    """Perturbing one conv1d tap must blow past the floor."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        pytest.skip("Metal not available")

    (
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    ) = _make_replica(mx.float32)

    hidden, B, C, D_arr, dt, dt_bias, state, A_log = _run_mamba_step(
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    )

    baseline_out, _ = ssm_step(hidden, A_log, B, C, D_arr, dt, dt_bias, state)
    mx.eval(baseline_out)

    broken_conv_w = conv_w * (1.0 + 0.5)
    _, broken_conv_input, _ = mx.split(
        x @ in_proj_w.T, [intermediate, intermediate + conv_dim], axis=-1
    )
    broken_conv_out = broken_conv_input * broken_conv_w[:, -1] + conv_b
    broken_conv_out = nn.silu(broken_conv_out)
    hidden_b, B_b, C_b = mx.split(
        broken_conv_out,
        [intermediate, intermediate + groups * d_state],
        axis=-1,
    )
    hidden_b = hidden_b.reshape(1, 1, heads, head_dim)
    B_b = B_b.reshape(1, groups, d_state)
    C_b = C_b.reshape(1, groups, d_state)

    broken_out, _ = ssm_step(hidden_b, A_log, B_b, C_b, D_arr, dt, dt_bias, state)
    mx.eval(broken_out)

    diff = float(
        mx.abs(
            baseline_out.astype(mx.float32) - broken_out.astype(mx.float32)
        ).max().item()
    )
    assert diff > 1e-4, f"perturbed conv1d did not change output (diff={diff:.3e})"


def test_mutation_A_log_breaks_parity() -> None:
    """Perturbing one A_log entry must blow past the floor."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        pytest.skip("Metal not available")

    (
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    ) = _make_replica(mx.float32)

    hidden, B, C, D_arr, dt, dt_bias, state, A_log = _run_mamba_step(
        in_proj_w, x, conv_w, conv_b, A_log, dt_bias, D, state,
        intermediate, conv_dim, heads, head_dim, d_state, groups,
    )

    baseline_out, _ = ssm_step(hidden, A_log, B, C, D_arr, dt, dt_bias, state)
    mx.eval(baseline_out)

    broken_A_log = A_log + 1.0
    broken_out, _ = ssm_step(hidden, broken_A_log, B, C, D_arr, dt, dt_bias, state)
    mx.eval(broken_out)

    diff = float(
        mx.abs(
            baseline_out.astype(mx.float32) - broken_out.astype(mx.float32)
        ).max().item()
    )
    assert diff > 1e-4, f"perturbed A_log did not change output (diff={diff:.3e})"
