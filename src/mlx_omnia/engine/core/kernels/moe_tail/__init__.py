"""The sorted-MoE prefill tail: one module per strategy, one delegator.

A prefill step declares the hidden width and `MoeTail` binds the specialization that
width admits, or none, at construction time. `sorted.py` reads the expert-sorted rows
through the permutation, which needs a hidden width that tiles by four columns per
thread; `default.py` serves everything else through the stock un-sort in ops, so the
delegator is total.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.moe_tail.default import DefaultMoeTail
from mlx_omnia.engine.core.kernels.moe_tail.kernel import MoeTailStrategy
from mlx_omnia.engine.core.kernels.moe_tail.sorted import SortedMoeTail
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = [
    "DefaultMoeTail",
    "MoeTail",
    "MoeTailStrategy",
    "SortedMoeTail",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (SortedMoeTail, DefaultMoeTail)


class MoeTail(MoeTailStrategy):
    """Resolves the strategy at construction and delegates; itself a
    `MoeTailStrategy`."""

    def __init__(self, *, hidden: int) -> None:
        self.strategy: MoeTailStrategy = resolve(_STRATEGIES, hidden=hidden)

    def __call__(
        self,
        sorted_expert_outputs: mx.array,
        inverse_order: mx.array,
        router_weights: mx.array,
        shared_output: mx.array,
        residual: mx.array,
        routed_scaling: float,
    ) -> mx.array:
        return self.strategy(
            sorted_expert_outputs,
            inverse_order,
            router_weights,
            shared_output,
            residual,
            routed_scaling,
        )
