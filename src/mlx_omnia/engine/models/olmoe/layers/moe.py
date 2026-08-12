import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import SORTED_GATHER_MIN, SwitchGLU, sorted_gather
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.olmoe.config import OlmoEConfig


class OlmoEMLP(nn.Module):
    def __init__(self, config: OlmoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.intermediate_size
        )
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        probs = softmax(self.gate(x), axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights

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
