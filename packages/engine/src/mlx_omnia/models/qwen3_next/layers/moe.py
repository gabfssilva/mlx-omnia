import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import SORTED_GATHER_MIN, SharedMLP, SwitchGLU, sorted_gather
from mlx_omnia.core.mxcompat import softmax
from mlx_omnia.models.qwen3_next.config import Qwen3NextConfig


class Qwen3NextMoE(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.shared_expert = SharedMLP(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob

    def __call__(self, x: mx.array) -> mx.array:
        probs = softmax(self.gate(x), axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        mixed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        return mixed + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
