import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import SORTED_GATHER_MIN, sorted_gather
from sideros.models.nemotron_h.config import NemotronHConfig
from sideros.models.nemotron_h.layers.mlp import NemotronHMLP, SwitchMLP


class NemotronHGate(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.routed_experts,))
        self.k = config.experts_per_tok
        self.n_group = config.expert_groups
        self.topk_group = config.expert_groups_kept
        self.scaling = config.routed_scale
        self.norm_topk = config.normalize_topk

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
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.scaling


class NemotronHMoE(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        latent = config.moe_latent_size
        self.gate = NemotronHGate(config)
        self.switch_mlp = SwitchMLP(
            config.routed_experts,
            latent if latent is not None else config.hidden_size,
            config.moe_inner,
        )
        if config.shared_expert_inner:
            self.shared_experts = NemotronHMLP(
                config.hidden_size, config.shared_expert_inner, config.mlp_bias
            )
        if latent is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, latent, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(latent, config.hidden_size, bias=config.mlp_bias)
        self.inner = latent if latent is not None else config.hidden_size
        self.k = config.experts_per_tok

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.gate(x)
        projected = self.fc1_latent_proj(x) if "fc1_latent_proj" in self else x
        length = projected.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(projected, chosen, k=self.k, hidden=self.inner, apply=apply)
        else:
            tokens = mx.expand_dims(projected, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        mixed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2).astype(x.dtype)
        if "fc2_latent_proj" in self:
            mixed = self.fc2_latent_proj(mixed)
        if "shared_experts" in self:
            return mixed + self.shared_experts(x)
        return mixed
