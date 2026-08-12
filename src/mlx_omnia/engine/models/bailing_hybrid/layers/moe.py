import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import (
    SORTED_GATHER_MIN,
    SharedMLP,
    SwitchGLU,
    sorted_gather,
    swish,
)
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig


class BailingHybridGate(nn.Module):
    """The router's two leaves under the checkpoint's own names."""

    def __init__(self, config: BailingHybridConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.expert_bias = mx.zeros((config.num_experts,))
        self.k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scaling = config.routed_scaling_factor
        self.norm_topk = config.norm_topk_prob

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.sigmoid(self.gate_proj(x).astype(mx.float32))
        selector = scores + self.expert_bias
        if self.n_group > 1:
            grouped = mx.unflatten(selector, axis=-1, shape=(self.n_group, -1))
            strength = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
            dropped = self.n_group - self.topk_group
            worst = mx.argpartition(strength, kth=dropped - 1, axis=-2)[..., :dropped, :]
            floor = mx.array(-mx.inf, mx.float32)
            selector = mx.flatten(mx.put_along_axis(grouped, worst, floor, axis=-2), -2, -1)
        chosen = mx.argpartition(-selector, kth=self.k - 1, axis=-1)[..., : self.k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.k > 1 and self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.scaling


class LimitedSwitchGLU(SwitchGLU):
    """The routed SwiGLU under a limit: `silu(gate)` capped above, `up` capped both ways."""

    def __init__(self, experts: int, hidden: int, inner: int, limit: float) -> None:
        super().__init__(experts, hidden, inner)
        self.limit = limit

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = mx.minimum(swish(pairs[..., 0]), self.limit)
        return gated * mx.clip(pairs[..., 1], -self.limit, self.limit)


class LimitedSharedMLP(SharedMLP):
    """`LimitedSwitchGLU`'s always-on sibling; the two limits are read from separate
    lists and do not have to agree on a layer."""

    def __init__(self, hidden: int, inner: int, limit: float) -> None:
        super().__init__(hidden, inner)
        self.limit = limit

    def __call__(self, x: mx.array) -> mx.array:
        gated = mx.minimum(swish(self.gate_proj(x)), self.limit)
        return self.down_proj(gated * mx.clip(self.up_proj(x), -self.limit, self.limit))


class BailingHybridMoE(nn.Module):
    def __init__(
        self, config: BailingHybridConfig, expert_limit: float, shared_limit: float
    ) -> None:
        super().__init__()
        self.gate = BailingHybridGate(config)
        self.switch_mlp = (
            LimitedSwitchGLU(
                config.num_experts, config.hidden_size, config.moe_intermediate_size, expert_limit
            )
            if expert_limit
            else SwitchGLU(
                config.num_experts, config.hidden_size, config.moe_intermediate_size
            )
        )
        self.shared_experts = (
            LimitedSharedMLP(config.hidden_size, config.shared_intermediate_size, shared_limit)
            if shared_limit
            else SharedMLP(config.hidden_size, config.shared_intermediate_size)
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
        return mixed + self.shared_experts(x)
