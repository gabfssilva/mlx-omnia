import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import SORTED_GATHER_MIN, SwitchLinear, sorted_gather
from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.longcat_flash_ngram.config import LongcatFlashNgramConfig


class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, config.total_experts, bias=config.router_bias
        )
        self.e_score_correction_bias = mx.zeros((config.total_experts,))
        self.k = config.moe_topk
        self.scale = config.routed_scaling_factor
        self.norm_topk = config.norm_topk_prob
        self.split = config.total_experts - self.k

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.classifier(x)
        scores = softmax(logits, axis=-1, precise=True)
        corrected = scores + self.e_score_correction_bias
        chosen = mx.argpartition(corrected, kth=-self.k, axis=-1)[..., -self.k :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk:
            denom = mx.sum(weights, axis=-1, keepdims=True) + 1e-20
            weights = weights / denom
        return chosen, weights * self.scale


class LongcatFlashExperts(nn.Module):
    """The 256 routed experts as two stacked ``SwitchLinear`` leaves.
    gate‖up is row-interleaved at load; identity experts have no weights."""

    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            config.n_routed_experts, config.hidden_size, 2 * config.expert_ffn_hidden_size
        )
        self.down_proj = SwitchLinear(
            config.n_routed_experts, config.expert_ffn_hidden_size, config.hidden_size
        )
        self.inner = config.expert_ffn_hidden_size

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False
    ) -> mx.array:
        fused = self.gate_up_proj(x, indices, sorted_indices=sorted_indices)
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gate = pairs[..., 0]
        activated = mx.sigmoid(gate) * gate * pairs[..., 1]
        return self.down_proj(activated, indices, sorted_indices=sorted_indices)


class LongcatFlashMoE(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.switch_mlp = LongcatFlashExperts(config)
        self.router = LongcatFlashTopkRouter(config)
        self.n_routed = config.n_routed_experts
        self.k = config.moe_topk
        self.hidden = config.hidden_size

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.router.route(x)
        identity_mask = chosen >= self.n_routed
        clamped = mx.where(identity_mask, 0, chosen)
        regular_weights = mx.where(identity_mask, 0.0, weights)

        length = x.shape[1]
        if length * self.k >= SORTED_GATHER_MIN:
            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(
                x, clamped, k=self.k, hidden=self.hidden, apply=apply
            )
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, clamped, sorted_indices=False).squeeze(-2)

        weighted = routed * mx.expand_dims(regular_weights, -1)
        final = mx.sum(weighted, axis=-2)
        identity_sum = mx.sum(
            mx.where(identity_mask, weights, 0.0), axis=-1, keepdims=True
        )
        return final + x * identity_sum
