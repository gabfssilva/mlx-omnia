"""The universal gate/up strategy: the leaf called as the layer it already is.

`build` accepts every leaf and every declaration, so it registers last and makes the
delegator total — `GateUp` always resolves, and a model uses it like any other layer.
The projection runs through the leaf's own `__call__` (mlx's gather kernels — dense
or quantized, every mode), then the declared epilogue in ops: the optional per-row
projection bias, the clamp-before-activation order, silu or swiglu_oai. What defines
it is universality, not the absence of a kernel; its cost is the op-boundary rounding
and the extra dispatches the specialized strategies fuse away.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from sideros.core.kernels.gate_up.kernel import Activation
from sideros.core.layers import QuantizedSwitchLinear, SwitchLinear


@dataclass(frozen=True)
class DefaultGateUp:
    leaf: SwitchLinear | QuantizedSwitchLinear
    activation: Activation
    limit: float | None
    bias: mx.array | None

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
    ) -> Self:
        return cls(leaf, activation, limit, bias)

    def __call__(self, row: mx.array, chosen: mx.array) -> mx.array:
        fused = self.leaf(row[None, None, None], chosen[None]).reshape(chosen.shape[0], -1)
        if self.bias is not None:
            fused = fused + self.bias[chosen]
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
