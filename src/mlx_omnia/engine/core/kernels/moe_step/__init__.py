"""The gate-free sparse block's T=1 step: one kernel module per format, one delegator.

A sparse block declares its two expert stacks and their widths, and `MoeStep` binds the
specialization that declaration admits, once, at construction. `nvfp4.py` serves the
nvfp4-quantized pair on the GPU; `default.py` serves everything else through the op
chain, so the delegator is total.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.moe_step.default import DefaultMoeStep, SharedPair
from mlx_omnia.engine.core.kernels.moe_step.kernel import MoeStepStrategy
from mlx_omnia.engine.core.kernels.moe_step.nvfp4 import Nvfp4MoeStep
from mlx_omnia.engine.core.kernels.resolve import resolve
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwitchLinear

__all__ = ["DefaultMoeStep", "MoeStep", "MoeStepStrategy", "Nvfp4MoeStep", "SharedPair"]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (Nvfp4MoeStep, DefaultMoeStep)


class MoeStep:
    """Resolves the strategy at construction and delegates; itself a
    `MoeStepStrategy`."""

    def __init__(
        self,
        *,
        fc1: SwitchLinear | QuantizedSwitchLinear,
        fc2: SwitchLinear | QuantizedSwitchLinear,
        hidden: int,
        inner: int,
        shared: SharedPair | None = None,
    ) -> None:
        self.strategy: MoeStepStrategy = resolve(
            _STRATEGIES,
            fc1=fc1,
            fc2=fc2,
            hidden=hidden,
            inner=inner,
            shared=shared,
        )

    def __call__(self, x: mx.array, chosen: mx.array, weights: mx.array) -> mx.array:
        return self.strategy(x, chosen, weights)
