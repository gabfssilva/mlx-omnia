import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import sigmoid_topk, softmax_topk_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchGLU,
    sorted_gather,
)
from sideros.models.laguna.config import LagunaConfig


class LagunaSparseMoe(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.shared_expert = SwiGLU(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.moe_routed_scaling_factor
        self.cap = config.moe_router_logit_softcapping

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Sigmoid routing (not softmax): scores are independent, selection adds the
        bias, weights come from the unbiased scores and renormalize."""
        logits = self.gate(x).astype(mx.float32)
        if self.cap > 0.0:
            logits = mx.tanh(logits / self.cap) * self.cap
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
            # The gemv kernels read an affine bias per group; MXFP carries none.
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                self.hidden, self.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            # The routing kernel skips the softcap; a capped config takes the op chain.
            and self.cap == 0.0
            and softmax_topk_applies(self.split + self.k, self.k)
        )

    def fused_step(self, h: mx.array, residual: mx.array) -> mx.array:
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        assert isinstance(gate_up, QuantizedSwitchLinear)
        assert isinstance(down, QuantizedSwitchLinear)
        assert gate_up.biases is not None and down.biases is not None
        chosen, weights = sigmoid_topk(
            self.gate(h).reshape(-1),
            self.e_score_correction_bias,
            self.k,
            scale=self.scaling,
        )
        act = moe_gate_up_act(
            h.reshape(-1),
            gate_up.weight,
            gate_up.scales,
            gate_up.biases,
            chosen,
            group_size=gate_up.group_size,
            bits=gate_up.bits,
        )
        # The shared expert quantizes at different bits than the routed stack, so
        # it can't ride the kernel's shared slot; it folds into the residual input
        # instead, and routed_scaling folds into the routing weights.
        combined = residual + self.shared_expert(h)
        return moe_down_combine(
            act.reshape(-1),
            down.weight,
            down.scales,
            down.biases,
            chosen,
            weights,
            combined.reshape(-1),
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
        return routed * self.scaling + self.shared_expert(x)
