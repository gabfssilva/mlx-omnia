"""The universal MoE tail: the stock combine, un-sort then weight, written in ops.

`build` accepts every shape, so it registers last and makes the delegator total. The
expert-sorted rows are gathered into token order — the full `tokens x top_k x hidden`
copy the specialized strategy reads through instead — and the router weights are applied
slot by slot in bf16, each product rounded before it accumulates; the total takes the
routed scaling factor, then the unrouted stack's output, then the residual. What defines
it is universality, not the absence of a kernel; its cost is the copy and the dispatch
per slot.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.kernels.moe_tail.kernel import MoeTailStrategy


@dataclass(frozen=True)
class DefaultMoeTail(MoeTailStrategy):
    hidden: int

    @classmethod
    def build(cls, *, hidden: int) -> Self:
        return cls(hidden)

    def __call__(
        self,
        sorted_expert_outputs: mx.array,
        inverse_order: mx.array,
        router_weights: mx.array,
        shared_output: mx.array,
        residual: mx.array,
        routed_scaling: float,
    ) -> mx.array:
        tokens, top_k = router_weights.shape[0], router_weights.shape[1]
        gathered = mx.take(
            sorted_expert_outputs, inverse_order.reshape(-1), axis=0
        ).reshape(tokens, top_k, self.hidden)
        weights = router_weights.astype(mx.bfloat16)
        total = mx.zeros((tokens, self.hidden), dtype=mx.bfloat16)
        for slot in range(top_k):
            total = gathered[:, slot] * weights[:, slot : slot + 1] + total
        scaled = total * mx.array(routed_scaling, dtype=mx.bfloat16)
        return residual + (scaled + shared_output)
