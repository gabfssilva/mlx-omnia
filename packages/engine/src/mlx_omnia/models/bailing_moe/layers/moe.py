import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import SORTED_GATHER_MIN, SharedMLP, SwitchGLU, sorted_gather
from mlx_omnia.core.mxcompat import softmax
from mlx_omnia.models.bailing_moe.config import BailingMoEConfig


class BailingMoEGate(nn.Module):
    def __init__(self, config: BailingMoEConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.num_experts, config.hidden_size))
        if config.moe_router_enable_expert_bias:
            self.expert_bias = mx.zeros((config.num_experts,))
        self.k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scaling = config.routed_scaling_factor
        self.norm_topk = config.norm_topk_prob
        self.sigmoid = config.sigmoid_router

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = (x @ self.weight.T).astype(mx.float32)
        scores = mx.sigmoid(logits) if self.sigmoid else softmax(logits, axis=-1, precise=True)
        selector = scores + self.expert_bias if "expert_bias" in self else scores
        if self.n_group > 1:
            grouped = mx.unflatten(selector, axis=-1, shape=(self.n_group, -1))
            strength = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
            dropped = self.n_group - self.topk_group
            worst = mx.argpartition(strength, kth=dropped - 1, axis=-2)[..., :dropped, :]
            selector = mx.flatten(
                mx.put_along_axis(grouped, worst, mx.array(0.0), axis=-2), -2, -1
            )
        chosen = mx.argpartition(selector, kth=-self.k, axis=-1)[..., -self.k :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.k > 1 and self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.scaling


class BailingMoEMLP(nn.Module):
    def __init__(self, config: BailingMoEConfig) -> None:
        super().__init__()
        self.gate = BailingMoEGate(config)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        if config.shared_intermediate_size:
            self.shared_experts = SharedMLP(
                config.hidden_size, config.shared_intermediate_size
            )
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.gate(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        mixed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2).astype(x.dtype)
        if "shared_experts" in self:
            return mixed + self.shared_experts(x)
        return mixed
