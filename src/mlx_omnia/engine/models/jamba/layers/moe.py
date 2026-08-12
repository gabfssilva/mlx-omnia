import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import SORTED_GATHER_MIN, SwitchGLU, sorted_gather
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.jamba.config import JambaConfig


class JambaMoE(nn.Module):
    """Top-k over the **raw** router logits, then a softmax over just those k."""

    def __init__(self, config: JambaConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.intermediate_size
        )
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.router(x)
        chosen = mx.argpartition(logits, kth=self.split, axis=-1)[..., self.split :]
        weights = softmax(mx.take_along_axis(logits, chosen, axis=-1), axis=-1, precise=True)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
