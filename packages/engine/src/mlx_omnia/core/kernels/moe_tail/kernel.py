"""The MoE tail's contract: what a strategy is and what a prefill step declares.

The primitive: the expert-sorted rows of a prefill step combined back into token
order — router weights, routed scaling, the unrouted stack's output and the residual —
(sorted_expert_outputs, inverse_order, router_weights, shared_output, residual,
routed_scaling) -> [tokens, hidden]. One module per strategy implements it; the
`MoeTail` delegator in `__init__.py` resolves which one serves a given shape, once,
at construction.
"""

from typing import Protocol

import mlx.core as mx


class MoeTailStrategy(Protocol):
    """The combine that closes a sorted-MoE prefill step: `sorted_expert_outputs`
    [tokens*top_k, hidden] bf16 in expert-sorted order, read through `inverse_order`
    [tokens, top_k] uint32, weighted by `router_weights` [tokens, top_k] fp32, scaled by
    `routed_scaling` and added to `shared_output` and `residual` (both [tokens, hidden]
    bf16) -> [tokens, hidden] bf16."""

    def __call__(
        self,
        sorted_expert_outputs: mx.array,
        inverse_order: mx.array,
        router_weights: mx.array,
        shared_output: mx.array,
        residual: mx.array,
        routed_scaling: float,
    ) -> mx.array: ...
