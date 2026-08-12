"""The universal down/combine strategy: the leaf called as the layer it already is.

`build` accepts every leaf and every declaration, so it registers last and makes the
delegator total. The projection runs through the leaf's own `__call__` (mlx's gather
kernels — dense or quantized, every mode), then the ops the specialized kernels fuse:
the optional per-row projection bias, the routing-weight multiply, the expert sum and
the residual add — preceded by the SwiGLU itself when the layout is `blocked` and the
gate/up half handed over its pair block un-activated. Unlike the affine kernel's spare
slot, the shared expert here is just its own layer call — any format serves. The cost is
the op-boundary rounding and the extra dispatches, which is why it registers last.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.down_combine.kernel import Layout
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear


@dataclass(frozen=True)
class DefaultDownCombine:
    leaf: SwitchLinear | QuantizedSwitchLinear
    bias: mx.array | None
    shared: nn.Linear | nn.QuantizedLinear | None
    layout: Layout

    @classmethod
    def build(
        cls,
        leaf: SwitchLinear | QuantizedSwitchLinear,
        *,
        hidden: int,
        inner: int,
        bias: mx.array | None,
        shared: nn.Linear | nn.QuantizedLinear | None,
        layout: Layout,
    ) -> Self:
        return cls(leaf, bias, shared, layout)

    def __call__(
        self, act: mx.array, chosen: mx.array, weights: mx.array, residual: mx.array
    ) -> mx.array:
        rows = act.reshape(chosen.shape[0], -1)
        if self.layout == "blocked":
            gate, up = mx.split(rows, 2, axis=-1)
            rows = gate * mx.sigmoid(gate) * up
        routed = chosen if self.shared is None else chosen[:-1]
        k = routed.shape[0]
        down = self.leaf(rows[None, :k, None], routed[None]).reshape(k, -1)
        if self.bias is not None:
            down = down + self.bias[routed]
        out = (down * weights[:k, None]).sum(axis=0)
        if self.shared is not None:
            out = out + weights[-1] * self.shared(rows[-1][None]).reshape(-1)
        return out + residual
