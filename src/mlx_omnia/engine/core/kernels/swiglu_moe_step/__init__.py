"""The gated sparse block's T=1 step: one arrangement per format family, one delegator.

`moe_step`'s SwiGLU sibling, with routing and the residual join inside the contract:
the packed strategy routes ordinally within its gate/up dispatch, so the router's
answer cannot arrive as an operand. `nvfp4_packed.py` serves the nvfp4 stack;
`default.py` composes the fine primitives (`Route`, `GateUp`, `DownCombine`) and
always builds, so the delegator is total.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.resolve import resolve
from mlx_omnia.engine.core.kernels.swiglu_moe_step.default import DefaultSwigluMoeStep
from mlx_omnia.engine.core.kernels.swiglu_moe_step.kernel import SwigluMoeStepStrategy
from mlx_omnia.engine.core.kernels.swiglu_moe_step.nvfp4_packed import (
    Nvfp4PackedSwigluMoeStep,
)
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwiGLU, SwitchLinear

__all__ = [
    "DefaultSwigluMoeStep",
    "Nvfp4PackedSwigluMoeStep",
    "SwigluMoeStep",
    "SwigluMoeStepStrategy",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (Nvfp4PackedSwigluMoeStep, DefaultSwigluMoeStep)


class SwigluMoeStep(SwigluMoeStepStrategy):
    """Resolves the strategy at construction and delegates; itself a
    `SwigluMoeStepStrategy`."""

    def __init__(
        self,
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
    ) -> None:
        self.strategy: SwigluMoeStepStrategy = resolve(
            _STRATEGIES,
            gate=gate,
            bias=bias,
            experts=experts,
            k=k,
            scale=scale,
            softcap=softcap,
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            hidden=hidden,
            inner=inner,
            shared=shared,
        )

    @property
    def worthwhile(self) -> bool:
        return self.strategy.worthwhile

    def __call__(
        self,
        x: mx.array,
        residual: mx.array,
        *,
        logits: mx.array | None = None,
        keys: mx.array | None = None,
    ) -> mx.array:
        return self.strategy(x, residual, logits=logits, keys=keys)

    def batch(self, h: mx.array, residual: mx.array) -> mx.array | None:
        return self.strategy.batch(h, residual)
