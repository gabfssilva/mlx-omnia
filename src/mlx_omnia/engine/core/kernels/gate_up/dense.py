"""The dense (unquantized) routed gate/up gemv of a one-token step.

`gather_mm` streams a T=1 routed expert MLP at ~450 GB/s on the M5 Max; a row-pair gemv
carrying the expert indirection itself sustains ~700 — the tile kernel dispatched at m=1
wastes half the bandwidth.

The stack is the checkpoint layout `[experts, 2*inner, hidden]`, rows contiguous, gate‖up
**block**-concatenated (`[w1 ‖ w3]` on the output axis, LFM2.5's load-time fusion) — not
the row interleave the quantized strategies want, which is what the `blocked` layout
declares. Under it the activation is not this half's: the pair block travels un-activated
to the down/combine half, whose dense kernel computes silu(gate)·up while streaming the
down weights, so nothing elementwise runs between the two dispatches.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.gate_up.kernel import (
    Activation,
    GateUpStrategy,
    Layout,
    OrdinalRouting,
)
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear
from mlx_omnia.engine.core.mxcompat import metal_kernel

_SOURCE = """
    uint pair = thread_position_in_grid.y;
    uint expert = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint n0 = pair * 2;
    uint n = (uint)N;
    uint kd = (uint)KD;
    uint k4 = kd / 4;
    if (n0 >= n) return;

    size_t base = (size_t)IDX[expert] * n * kd;
    const device vec<T, 4>* w0 = (const device vec<T, 4>*)(W + base + (size_t)n0 * kd);
    const device vec<T, 4>* w1 = (const device vec<T, 4>*)(W + base + (size_t)(n0 + 1) * kd);
    const device vec<T, 4>* xv = (const device vec<T, 4>*)X;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint i = lane; i < k4; i += 32) {
        float4 xm = float4(xv[i]);
        acc0 += xm * float4(w0[i]);
        acc1 += xm * float4(w1[i]);
    }
    float t0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float t1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0) {
        Y[(size_t)expert * n + n0] = (T)t0;
        Y[(size_t)expert * n + n0 + 1] = (T)t1;
    }
"""

_KERNEL = metal_kernel(
    name="moe_dense_gate_up",
    input_names=["X", "W", "IDX", "N", "KD"],
    output_names=["Y"],
    source=_SOURCE,
)


def applies(hidden: int) -> bool:
    """The contraction is a float4 lane sweep; the output rows are walked in pairs, and
    `2*inner` is even for every `inner`."""
    return hidden % 4 == 0


@dataclass(frozen=True)
class DenseGateUp(GateUpStrategy):
    weight: mx.array

    @classmethod
    def build(
        cls,
        leaf: SwitchLinear | QuantizedSwitchLinear,
        *,
        hidden: int,
        inner: int,
        activation: Activation,
        limit: float | None,
        bias: mx.array | None,
        layout: Layout,
        routing: OrdinalRouting | None,
        shared: nn.Linear | nn.QuantizedLinear | None,
    ) -> Self | None:
        # The spare slot is the affine kernel's; every other format runs the routed
        # stack alone, so a shared expert beside it has nowhere to go.
        if shared is not None:
            return None
        if not isinstance(leaf, SwitchLinear) or layout != "blocked" or routing is not None:
            return None
        if activation != "silu" or limit is not None or bias is not None:
            return None
        if leaf.weight.shape[1:] != (2 * inner, hidden) or not applies(hidden):
            return None
        return cls(leaf.weight)

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        rows = self.weight.shape[1]
        kdim = self.weight.shape[2]
        assert kdim % 4 == 0 and rows % 2 == 0
        return _KERNEL(
            inputs=[
                row, self.weight, chosen,
                mx.array(rows, dtype=mx.int32),
                mx.array(kdim, dtype=mx.int32),
            ],
            template=[("T", row.dtype)],
            grid=(32, rows // 2, chosen.shape[0]),
            threadgroup=(32, 1, 1),
            output_shapes=[(chosen.shape[0], rows)],
            output_dtypes=[row.dtype],
        )[0]
