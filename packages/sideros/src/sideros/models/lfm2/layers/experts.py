import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.moe_gemv_dense import (
    moe_dense_down,
    moe_dense_gate_up,
    moe_gemv_dense_applies,
    moe_route_sigmoid,
)
from sideros.core.layers import SORTED_GATHER_MIN, SwitchLinear, sorted_gather
from sideros.models.lfm2.config import LFM2MoEConfig
from sideros.models.lfm2.layers import flags


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

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.sigmoid(self.gate(x))
        selector = scores.astype(mx.float32) + self.expert_bias if "expert_bias" in self else scores
        chosen = mx.argpartition(selector, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-6)
        return chosen, weights * self.scaling

    def fused_step_applies(self) -> bool:
        return flags.MOE_GEMV_DENSE_FUSED and moe_gemv_dense_applies(
            self.hidden, self.experts.inner, self.experts.w13.weight.shape[0], self.k
        )

    def fused_step(self, x: mx.array) -> mx.array:
        """Routing, both gemvs, silu and the routing weight in three dispatches."""
        row = x.reshape(-1)
        gate = self.gate.weight
        assert isinstance(gate, mx.array)
        if flags.MOE_ROUTE_FUSED:
            bias = (
                self.expert_bias
                if "expert_bias" in self
                else mx.zeros(gate.shape[0], mx.float32)
            )
            assert isinstance(bias, mx.array)
            chosen, weights = moe_route_sigmoid(
                row,
                gate,
                bias,
                mx.array(self.scaling, mx.float32),
                self.k,
                normalized=self.norm_topk,
            )
        else:
            selected, scores = self.route(x)
            chosen, weights = selected.reshape(-1).astype(mx.uint32), scores.reshape(-1)
        gate_up = moe_dense_gate_up(row, self.experts.w13.weight, chosen)
        routed = moe_dense_down(gate_up, self.experts.w2.weight, chosen, weights)
        return routed.sum(axis=0).reshape(x.shape)

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
