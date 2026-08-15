"""The composed chain for the gated T=1 sparse step; always builds.

Not ops all the way down: the three fine primitives resolve on their own, so an affine
checkpoint still lands its affine kernels through this strategy. What is fixed here is
the arrangement the fused strategies must reproduce — the shared expert folded into the
residual input (its format owes the routed stack nothing) and the routed scaling folded
into the routing weights.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.down_combine import DownCombine
from mlx_omnia.engine.core.kernels.gate_up import GateUp
from mlx_omnia.engine.core.kernels.route import DefaultRoute, Route
from mlx_omnia.engine.core.kernels.swiglu_moe_step.kernel import SwigluMoeStepStrategy
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwiGLU, SwitchLinear


@dataclass(frozen=True)
class DefaultSwigluMoeStep(SwigluMoeStepStrategy):
    route: Route
    gate_up: GateUp
    down: DownCombine
    shared: SwiGLU

    @classmethod
    def build(
        cls,
        *,
        gate: mx.array,
        bias: mx.array,
        experts: int,
        k: int,
        scale: float,
        softcap: float,
        gate_up_proj: SwitchLinear | QuantizedSwitchLinear,
        down_proj: SwitchLinear | QuantizedSwitchLinear,
        hidden: int,
        inner: int,
        shared: SwiGLU,
    ) -> Self:
        return cls(
            Route(
                gate,
                experts=experts,
                k=k,
                scoring="sigmoid",
                bias=bias,
                scale=scale,
                softcap=softcap,
            ),
            GateUp(gate_up_proj, hidden=hidden, inner=inner),
            DownCombine(down_proj, hidden=hidden, inner=inner),
            shared,
        )

    @property
    def worthwhile(self) -> bool:
        return not isinstance(self.route.strategy, DefaultRoute)

    def __call__(
        self,
        x: mx.array,
        residual: mx.array,
        *,
        logits: mx.array | None = None,
        keys: mx.array | None = None,
    ) -> mx.array:
        row = x.reshape(-1)
        chosen, weights = self.route(
            row, logits=None if logits is None else logits.reshape(-1)
        )
        combined = residual + self.shared(x)
        act = self.gate_up(row, chosen)
        return self.down(act, chosen, weights, combined.reshape(-1)).reshape(x.shape)

    def batch(self, h: mx.array, residual: mx.array) -> mx.array | None:
        return None
