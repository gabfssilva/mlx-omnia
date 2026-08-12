import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import (
    SORTED_GATHER_MIN,
    SharedMLP,
    SwitchGLU,
    sorted_gather,
)
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.ernie4_5_moe.config import Ernie45MoEConfig


class Ernie45MoEMLP(nn.Module):
    def __init__(self, config: Ernie45MoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.moe_num_experts, config.hidden_size, config.expert_intermediate_size
        )
        if config.shared_intermediate_size:
            self.shared_experts = SharedMLP(
                config.hidden_size, config.shared_intermediate_size
            )
        self.hidden = config.hidden_size
        self.k = config.moe_k
        self.split = config.moe_num_experts - self.k
        self.sigmoid = config.sigmoid_gate

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.gate(x).astype(mx.float32)
        scores = mx.sigmoid(logits) if self.sigmoid else softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(scores, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        return chosen, weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)

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
        mixed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2).astype(x.dtype)
        if "shared_experts" in self:
            return mixed + self.shared_experts(x)
        return mixed
