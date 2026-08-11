import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import SORTED_GATHER_MIN, SharedMLP, SwitchGLU, sorted_gather
from mlx_omnia.core.mxcompat import softmax
from mlx_omnia.models.deepseek_v2.config import DeepseekV2Config


class DeepseekV2Gate(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.k = config.num_experts_per_tok
        self.n_group = config.groups
        self.topk_group = config.topk_groups
        self.scaling = config.routed_scaling_factor
        self.group_limited = config.group_limited

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = softmax(x @ self.weight.T, axis=-1, precise=True)
        if self.group_limited:
            grouped = mx.unflatten(scores, axis=-1, shape=(self.n_group, -1))
            strength = grouped.max(axis=-1, keepdims=True)
            dropped = self.n_group - self.topk_group
            worst = mx.argpartition(strength, kth=dropped - 1, axis=-2)[..., :dropped, :]
            scores = mx.flatten(
                mx.put_along_axis(grouped, worst, mx.array(0.0, scores.dtype), axis=-2), -2, -1
            )
        chosen = mx.argpartition(-scores, kth=self.k - 1, axis=-1)[..., : self.k]
        return chosen, mx.take_along_axis(scores, chosen, axis=-1) * self.scaling


class DeepseekV2MoE(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        self.gate = DeepseekV2Gate(config)
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
        mixed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        if "shared_experts" in self:
            return mixed + self.shared_experts(x)
        return mixed
