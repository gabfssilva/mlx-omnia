"""The routed MLP, on every layer of the sparse trunk.

The T=1 step routes through the fused kernels when the leaves' formats and shapes
serve them; prefill gathers sorted by expert — a pure reorder.
"""

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.down_combine import DownCombine
from sideros.core.kernels.gate_up import GateUp
from sideros.core.kernels.moe_route import softmax_topk, softmax_topk_applies
from sideros.core.layers import SORTED_GATHER_MIN, SwitchGLU, sorted_gather
from sideros.core.mxcompat import softmax
from sideros.models.qwen3.config import Qwen3MoEConfig


class Qwen3MoEMLP(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The softmax spans all experts, so the kept weights depend on the dropped
        ones; renormalizing after the cut matches transformers' sorted topk."""
        probs = softmax(self.gate(x), axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights

    def _kernels(self) -> tuple[GateUp, DownCombine]:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final."""
        gate_up, down = self._gate_up, self._down
        if gate_up is None or down is None:
            switch = self.switch_mlp
            gate_up = GateUp(switch.gate_up_proj, hidden=self.hidden, inner=switch.inner)
            down = DownCombine(switch.down_proj, hidden=self.hidden, inner=switch.inner)
            self._gate_up, self._down = gate_up, down
        return gate_up, down

    def step_applies(self) -> bool:
        return self.norm_topk and softmax_topk_applies(self.split + self.k, self.k)

    def step(self, h: mx.array, residual: mx.array) -> mx.array:
        """T=1: routing, both expert gemvs, the routed sum and the residual join in
        three dispatches."""
        gate_up, down = self._kernels()
        chosen, weights = softmax_topk(self.gate(h).reshape(-1), self.k)
        act = gate_up(h.reshape(-1), chosen)
        return down(act, chosen, weights, residual.reshape(-1)).reshape(1, 1, self.hidden)

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
