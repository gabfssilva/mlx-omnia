"""The universal routing strategy: the declaration spelled out in ops.

`build` accepts every router and every declaration, so it registers last and makes the
delegator total — `Route` always resolves. The chain is the one the model families
already run when no kernel applies: the gate gemv, the optional softcap, the scoring,
`argpartition` on the biased scores with `take_along_axis` behind it, the
renormalization and the scale. Selection uses the ascending partition (`kth = experts -
k`, keep the tail), so an exact tie goes to the higher index — the rule every kernel
here but `OrdinalRoute` follows. What defines it is universality, not the absence of a
kernel; its cost is the ~10 tiny dependent dispatches the specialized strategies fuse
into one.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.route.kernel import Routing, gate_logits
from mlx_omnia.engine.core.mxcompat import softmax


@dataclass(frozen=True)
class DefaultRoute:
    gate: mx.array | None
    routing: Routing

    @classmethod
    def build(cls, gate: mx.array | None, *, routing: Routing) -> Self:
        return cls(gate, routing)

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        routing = self.routing
        values = gate_logits(self.gate, row, logits)
        if routing.softcap > 0.0:
            values = mx.tanh(values / routing.softcap) * routing.softcap
        routed = values[: routing.experts]
        if routing.scoring == "softmax":
            scores = softmax(routed, axis=-1, precise=True)
        elif routing.scoring == "sigmoid":
            scores = mx.sigmoid(routed)
        else:
            scores = mx.sqrt(nn.softplus(routed.astype(mx.float32)))

        table = routing.hash_table
        if table is not None:
            assert ids is not None
            chosen = table[ids].reshape(-1).astype(mx.uint32)
        else:
            biased = scores if routing.bias is None else scores + routing.bias
            split = routing.experts - routing.k
            chosen = mx.argpartition(biased, kth=split, axis=-1)[split:].astype(mx.uint32)

        weights = scores[chosen]
        if routing.normalize:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + routing.norm_eps)
        weights = weights * routing.scale
        if routing.shared:
            slot = mx.array([routing.experts], dtype=mx.uint32)
            shared = mx.sigmoid(values[routing.experts :]).astype(weights.dtype)
            return mx.concatenate([chosen, slot]), mx.concatenate([weights, shared])
        return chosen, weights
