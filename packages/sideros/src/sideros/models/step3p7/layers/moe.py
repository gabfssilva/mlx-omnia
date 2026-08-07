import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import SORTED_GATHER_MIN, sorted_gather
from sideros.models.step3p7.config import Step3p7TextConfig
from sideros.models.step3p7.layers.mlp import Step3p7MLP, Step3p7SwitchGLU


class Step3p7MoE(nn.Module):
    def __init__(self, config: Step3p7TextConfig, layer: int) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        if config.use_moe_router_bias:
            self.router_bias = mx.zeros((config.moe_num_experts,), dtype=mx.float32)
        self.switch_mlp = Step3p7SwitchGLU(
            config.moe_num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            config.limits[layer],
            config.limits[layer],
        )
        self.shared_expert = Step3p7MLP(
            config.hidden_size,
            config.share_expert_dim,
            config.shared_limits[layer],
            config.shared_limits[layer],
        )
        self.k = config.moe_top_k
        self.split = config.moe_num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.moe_router_scaling_factor
        self.need_fp32_gate = config.need_fp32_gate
        self.norm_expert_weight = config.norm_expert_weight

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.gate(x.astype(mx.float32)) if self.need_fp32_gate else self.gate(x)
        scores = mx.sigmoid(logits)
        bias = self.router_bias if "router_bias" in self else mx.zeros_like(scores)
        biased = scores + bias
        chosen = mx.argpartition(biased, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_expert_weight:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights * self.scaling

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
        routed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        return routed + self.shared_expert(x)
