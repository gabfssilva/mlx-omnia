import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import SORTED_GATHER_MIN, SwitchLinear, sorted_gather
from sideros.models.deepseek_v4.config import DeepseekV4Config


def clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """GPT-OSS's clamp order — gate capped above, up clamped both ways, *before* the
    activation — with plain silu."""
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return gate * mx.sigmoid(gate) * up


class MoEGate(nn.Module):
    """`sqrt(softplus)` over the router logits in fp32; the selection adds an fp32 bias the
    kept weights never see. The first `num_hash_layers` layers do not select at all — the
    experts are a table lookup on the token id."""

    def __init__(self, config: DeepseekV4Config, layer: int) -> None:
        super().__init__()
        self.k = config.num_experts_per_tok
        self.norm_topk = config.norm_topk_prob
        self.scaling = config.routed_scaling_factor
        self.hash = layer < config.num_hash_layers
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (config.n_routed_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, ids: mx.array) -> tuple[mx.array, mx.array]:
        scores = mx.sqrt(nn.softplus((x @ self.weight.T).astype(mx.float32)))
        if self.hash:
            chosen = self.tid2eid[ids]
        else:
            biased = scores + self.e_score_correction_bias
            chosen = mx.argpartition(-biased, kth=self.k - 1, axis=-1)[..., : self.k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.scaling


class ClampedSwiGLU(nn.Module):
    """The shared expert. gate‖up concatenated on the output axis at load."""

    def __init__(self, hidden: int, inner: int, limit: float) -> None:
        super().__init__()
        self.inner = inner
        self.limit = limit
        self.gate_up_proj = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), [self.inner], axis=-1)
        return self.down_proj(clamped_swiglu(gate, up, self.limit))


class SwitchGLU(nn.Module):
    """The routed experts, gate and up row-interleaved ([g0,u0,g1,u1,…]) at load: one
    gather reads both."""

    def __init__(self, experts: int, hidden: int, inner: int, limit: float) -> None:
        super().__init__()
        self.inner = inner
        self.limit = limit
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        fused = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        activated = clamped_swiglu(pairs[..., 0], pairs[..., 1], self.limit)
        return self.down_proj(activated, indices, sorted_indices=sorted_indices)


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer: int) -> None:
        super().__init__()
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.gate = MoEGate(config, layer)
        self.switch_mlp = SwitchGLU(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            config.swiglu_limit,
        )
        self.shared_experts = ClampedSwiGLU(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.swiglu_limit,
        )

    def __call__(self, x: mx.array, ids: mx.array) -> mx.array:
        chosen, weights = self.gate(x, ids)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        combined = (routed * mx.expand_dims(weights, -1).astype(routed.dtype)).sum(axis=-2)
        return combined + self.shared_experts(x)
