"""The routed MLP, on every layer of the sparse trunk.

The T=1 step routes through the fused kernels when the shapes tile (`moe_gemv_applies`);
prefill gathers sorted by expert — a pure reorder.
"""

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import softmax_topk, softmax_topk_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwitchGLU,
    sorted_gather,
)
from sideros.core.mxcompat import softmax
from sideros.models.qwen3.config import Qwen3MoEConfig


class Qwen3MoEMLP(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.norm_topk = config.norm_topk_prob

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The softmax spans all experts, so the kept weights depend on the dropped
        ones; renormalizing after the cut matches transformers' sorted topk."""
        probs = softmax(self.gate(x), axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        if self.norm_topk:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights

    def step_applies(self) -> bool:
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        return (
            self.norm_topk
            and isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            # The gemv kernels read an affine bias per group; MXFP carries none.
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                self.hidden, self.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(self.split + self.k, self.k)
        )

    def step(self, h: mx.array, residual: mx.array) -> mx.array:
        """T=1: routing, both expert gemvs, the routed sum and the residual join in
        three dispatches."""
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        assert isinstance(gate_up, QuantizedSwitchLinear)
        assert isinstance(down, QuantizedSwitchLinear)
        assert gate_up.biases is not None and down.biases is not None
        chosen, weights = softmax_topk(self.gate(h).reshape(-1), self.k)
        act = moe_gate_up_act(
            h.reshape(-1),
            gate_up.weight,
            gate_up.scales,
            gate_up.biases,
            chosen,
            group_size=gate_up.group_size,
            bits=gate_up.bits,
        )
        return moe_down_combine(
            act.reshape(-1),
            down.weight,
            down.scales,
            down.biases,
            chosen,
            weights,
            residual.reshape(-1),
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
        return (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
