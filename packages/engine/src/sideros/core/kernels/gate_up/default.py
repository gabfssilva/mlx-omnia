"""The universal gate/up strategy: the leaf called as the layer it already is.

`build` accepts every leaf and every declaration, so it registers last and makes the
delegator total — `GateUp` always resolves, and a model uses it like any other layer.
The projection runs through the leaf's own `__call__` (mlx's gather kernels — dense
or quantized, every mode), then the declared epilogue in ops: the optional per-row
projection bias, the clamp-before-activation order, silu or swiglu_oai. Under a `blocked`
layout the epilogue is the down/combine half's, so the pair block comes back as it is; an
`OrdinalRouting` declaration is honoured by resolving the keys to indices first, and a
declared shared expert is its own layer call on the last slot, so any format serves. What
defines it is universality, not the absence of a kernel; its cost is the op-boundary
rounding and the extra dispatches the specialized strategies fuse away.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.gate_up.kernel import Activation, Layout, OrdinalRouting
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear


@dataclass(frozen=True)
class DefaultGateUp:
    leaf: SwitchLinear | QuantizedSwitchLinear
    activation: Activation
    limit: float | None
    bias: mx.array | None
    layout: Layout
    routing: OrdinalRouting | None
    shared: nn.Linear | nn.QuantizedLinear | None

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
    ) -> Self:
        # The last slot is the shared expert's, so the router's keys cannot also decide
        # it — and this strategy has to stay total, so the pair is refused here.
        assert shared is None or routing is None
        return cls(leaf, activation, limit, bias, layout, routing, shared)

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        if self.routing is not None:
            chosen = self.routing.indices(chosen)
        routed = chosen if self.shared is None else chosen[:-1]
        fused = self.leaf(row[None, None, None], routed[None]).reshape(routed.shape[0], -1)
        if self.bias is not None:
            fused = fused + self.bias[routed]
        if self.shared is not None:
            # Its own layer call, like the down half's spare: any format serves.
            fused = mx.concatenate([fused, self.shared(row.reshape(1, -1))], axis=0)
        if self.layout == "blocked":
            return fused
        pairs = fused.reshape(chosen.shape[0], -1, 2)
        gate, up = pairs[..., 0], pairs[..., 1]
        if self.activation == "swiglu_oai":
            assert self.limit is not None
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
            return gate * mx.sigmoid(1.702 * gate) * (up + 1)
        if self.limit is not None:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return gate * mx.sigmoid(gate) * up
