import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import sigmoid_topk, softmax_topk_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchLinear,
    sorted_gather,
)
from sideros.models.hy3.config import Hy3Config


class Hy3SwitchGLU(nn.Module):
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

    def __call__(
        self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class Hy3SparseMoe(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.switch_mlp = Hy3SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
        self.shared_expert = SwiGLU(config.hidden_size, shared_intermediate)
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.router_scaling_factor
        self.fp32_combine = config.enable_moe_fp32_combine

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Sigmoid routing (not softmax): scores are independent, selection adds the
        bias, weights come from the unbiased scores and renormalize. The router gemv
        runs in fp32 to match transformers' `F.linear(x.float(), w.float())`."""
        logits = self.gate(x.astype(mx.float32)).astype(mx.float32)
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias.astype(scores.dtype)
        chosen = mx.argpartition(biased, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights.astype(x.dtype)

    def fused_step_applies(self) -> bool:
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                self.hidden, self.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(self.split + self.k, self.k)
            and not self.fp32_combine
        )

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array:
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        assert isinstance(gate_up, QuantizedSwitchLinear)
        assert isinstance(down, QuantizedSwitchLinear)
        assert gate_up.biases is not None and down.biases is not None
        chosen, weights = sigmoid_topk(
            self.gate(x.astype(mx.float32)).astype(x.dtype).reshape(-1),
            self.e_score_correction_bias,
            self.k,
            scale=self.scaling,
        )
        act = moe_gate_up_act(
            x.reshape(-1),
            gate_up.weight,
            gate_up.scales,
            gate_up.biases,
            chosen,
            group_size=gate_up.group_size,
            bits=gate_up.bits,
        )
        joined = residual + self.shared_expert(x)
        return moe_down_combine(
            act.reshape(-1),
            down.weight,
            down.scales,
            down.biases,
            chosen,
            weights,
            joined.reshape(-1),
            group_size=down.group_size,
            bits=down.bits,
        ).reshape(1, 1, self.hidden)

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
        routed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        routed = routed * self.scaling
        shared = self.shared_expert(x)
        if self.fp32_combine:
            return (routed.astype(mx.float32) + shared.astype(mx.float32)).astype(x.dtype)
        return routed + shared
