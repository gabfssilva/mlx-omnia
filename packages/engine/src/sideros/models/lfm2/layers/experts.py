import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.down_combine import DownCombine
from sideros.core.kernels.gate_up import GateUp
from sideros.core.kernels.route import Route
from sideros.core.layers import SORTED_GATHER_MIN, SwitchLinear, sorted_gather
from sideros.models.lfm2.config import LFM2MoEConfig

NORM_EPS = 1e-6
"""transformers renormalizes the kept sigmoid scores against this floor; it is part of
the routing declaration, not a guard, so both paths read the same constant."""


class LFM2Experts(nn.Module):
    """Gate and up block-concatenated ([w1 ‖ w3] on the output axis): read by slice."""

    def __init__(self, count: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.w13 = SwitchLinear(count, hidden, 2 * inner)
        self.w2 = SwitchLinear(count, inner, hidden)
        self.inner = inner

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        fused = self.w13(tokens, indices, sorted_indices=sorted_indices)
        gated = fused[..., : self.inner]
        activated = gated * mx.sigmoid(gated) * fused[..., self.inner :]
        return self.w2(activated, indices, sorted_indices=sorted_indices)


class LFM2SparseMLP(nn.Module):
    """32 experts, 4 per token, sigmoid-routed: the float32 `expert_bias` shifts which
    experts win but never their weights, which come from the bias-free score."""

    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = LFM2Experts(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        if config.use_expert_bias:
            self.expert_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob
        self.scaling = config.routed_scaling_factor
        self._route: Route | None = None
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.sigmoid(self.gate(x))
        selector = scores.astype(mx.float32) + self.expert_bias if "expert_bias" in self else scores
        chosen = mx.argpartition(selector, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + NORM_EPS)
        return chosen, weights * self.scaling

    def fused_step_applies(self) -> bool:
        """The routing primitive contracts on the router's matrix, which only a dense
        gate leaf carries; the two gemv primitives are total over the leaf formats."""
        return type(self.gate) is nn.Linear

    def kernels(self) -> tuple[Route, GateUp, DownCombine]:
        """Resolved once, at the first T=1 step — after load, when the leaves' formats
        are final."""
        route, gate_up, down = self._route, self._gate_up, self._down
        if route is None or gate_up is None or down is None:
            gate = self.gate.weight
            assert isinstance(gate, mx.array)
            experts = self.experts.w13.weight.shape[0]
            bias = (
                self.expert_bias
                if "expert_bias" in self
                else mx.zeros((experts,), mx.float32)
            )
            assert isinstance(bias, mx.array)
            route = Route(
                gate,
                experts=experts,
                k=self.k,
                scoring="sigmoid",
                bias=bias,
                normalize=self.norm_topk,
                norm_eps=NORM_EPS if self.norm_topk else 0.0,
                scale=self.scaling,
            )
            gate_up = GateUp(
                self.experts.w13,
                hidden=self.hidden,
                inner=self.experts.inner,
                layout="blocked",
            )
            down = DownCombine(
                self.experts.w2,
                hidden=self.hidden,
                inner=self.experts.inner,
                layout="blocked",
            )
            self._route, self._gate_up, self._down = route, gate_up, down
        return route, gate_up, down

    def fused_step(self, x: mx.array) -> mx.array:
        """Routing, both gemvs, silu and the routing weight in three dispatches."""
        route, gate_up, down = self.kernels()
        row = x.reshape(-1)
        chosen, weights = route(row)
        act = gate_up(row, chosen)
        return down(act, chosen, weights, mx.zeros_like(row)).reshape(x.shape)

    def __call__(self, x: mx.array) -> mx.array:
        if x.size == self.hidden and self.fused_step_applies():
            return self.fused_step(x)
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.experts(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.experts(tokens, chosen, sorted_indices=False).squeeze(-2)
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
