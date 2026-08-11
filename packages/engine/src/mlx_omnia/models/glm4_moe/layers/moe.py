import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import (
    SORTED_GATHER_MIN,
    SharedMLP,
    SwitchGLU,
    sorted_gather,
)
from mlx_omnia.models.glm4_moe.config import Glm4MoEConfig


class Glm4MoEGate(nn.Module):
    """The router's two leaves under the checkpoint's own names."""

    def __init__(self, config: Glm4MoEConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scaling = config.routed_scaling_factor
        self.norm_topk = config.norm_topk_prob

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        selector = scores + self.e_score_correction_bias
        if self.n_group > 1:
            grouped = mx.unflatten(selector, axis=-1, shape=(self.n_group, -1))
            strength = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
            dropped = self.n_group - self.topk_group
            worst = mx.argpartition(strength, kth=dropped - 1, axis=-2)[..., :dropped, :]
            selector = mx.flatten(
                mx.put_along_axis(grouped, worst, mx.array(0.0), axis=-2), -2, -1
            )
        chosen = mx.argpartition(-selector, kth=self.k - 1, axis=-1)[..., : self.k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.k > 1 and self.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights * self.scaling


class Glm4MoEMLP(nn.Module):
    def __init__(self, config: Glm4MoEConfig) -> None:
        super().__init__()
        self.gate = Glm4MoEGate(config)
        self.switch_mlp = SwitchGLU(
            config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
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
