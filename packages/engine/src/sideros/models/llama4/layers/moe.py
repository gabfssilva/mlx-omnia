import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import SORTED_GATHER_MIN, SwiGLU, SwitchLinear, sorted_gather
from sideros.models.llama4.config import Llama4TextConfig


class Llama4SwitchGLU(nn.Module):
    """Gate and up fused row-interleaved ([g0,u0,g1,u1,…) at load: one gather reads both."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class Llama4MoE(nn.Module):
    """Sigmoid top-1 routing with pre-multiplication and an ungated shared expert.

    The sigmoid score pre-multiplies the expert input (transformers
    authoritative): `expert(x * sigmoid(logit))`. The shared expert is always on
    (weight 1.0, no gate) and added to the routed output.
    """

    def __init__(self, config: Llama4TextConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.switch_mlp = Llama4SwitchGLU(
            config.num_local_experts, config.hidden_size, config.intermediate_size
        )
        self.shared_expert = SwiGLU(config.hidden_size, config.intermediate_size)
        self.k = config.num_experts_per_tok
        self.hidden = config.hidden_size

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Argmax top-1, sigmoid in fp32 then cast to dtype. No renorm, no bias."""
        logits = self.gate(x)
        chosen = mx.argmax(logits, axis=-1)[..., None]
        scores = mx.take_along_axis(logits, chosen, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)
        return chosen, scores

    def __call__(self, x: mx.array) -> mx.array:
        chosen, scores = self.route(x)
        x_scaled = x * scores
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x_scaled, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x_scaled, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        return routed.squeeze(-2) + self.shared_expert(x)
