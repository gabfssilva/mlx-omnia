from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import softmax_topk, softmax_topk_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwitchLinear,
    sorted_gather,
)
from sideros.core.mxcompat import softmax
from sideros.models.qwen3_5.config import Qwen35TextConfig


def _packed(linear: nn.QuantizedLinear) -> tuple[mx.array, mx.array, mx.array]:
    """A quantized leaf's three tensors, narrowed: `biases` is optional on the type but
    never absent under affine quantization."""
    assert isinstance(linear.biases, mx.array)
    return linear.weight, linear.scales, linear.biases


class Qwen35SharedExpert(nn.Module):
    """Only the shared expert's `down_proj` lives here: its gate and up are row 256 of
    the routed stack, but appending a 257th row to the *down* stack would materialize
    the one tensor that goes mmap'd from the file into the model (5.4 GB at 6 bits)."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)


class Qwen35SwitchGLU(nn.Module):
    """gate‖up row-interleaved ([g0,u0,g1,…]) with the shared expert stacked as the
    last row — one gather reads both projections of any slot, routed or shared."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts + 1, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]


class Qwen35MoEInternals(NamedTuple):
    probs: mx.array
    indices: mx.array
    weights: mx.array
    routed: mx.array
    shared: mx.array
    out: mx.array


class Qwen35MoE(nn.Module):
    """256 experts, 8 per token, plus a shared expert scaled by the sigmoid of its own
    logit. That logit is row 256 of the router's matrix, so one gemv produces both."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.experts = config.num_experts
        self.k = config.num_experts_per_tok
        self.hidden = config.hidden_size
        self.gate = nn.Linear(config.hidden_size, config.num_experts + 1, bias=False)
        self.switch_mlp = Qwen35SwitchGLU(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.shared_expert = Qwen35SharedExpert(config)

    def route(self, logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """The softmax spans all 256 experts, so the kept weights depend on the dropped
        ones; renormalizing after the cut matches transformers' sorted top-k."""
        probs = softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(probs, kth=self.experts - self.k, axis=-1)[..., -self.k :]
        weights = mx.take_along_axis(probs, chosen, axis=-1)
        return probs, chosen, weights / weights.sum(axis=-1, keepdims=True)

    def _routed(self, x: mx.array, chosen: mx.array) -> mx.array:
        """[1, T, k, hidden] — one row per kept expert, unweighted."""
        switch = self.switch_mlp
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                act = switch.activate(switch.gate_up_proj(tokens, experts, sorted_indices=True))
                return switch.down_proj(act, experts, sorted_indices=True)

            return sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        tokens = mx.expand_dims(x, (-2, -3))
        act = switch.activate(switch.gate_up_proj(tokens, chosen))
        return switch.down_proj(act, chosen).squeeze(-2)

    def _shared(self, x: mx.array) -> mx.array:
        """The shared expert reads slot 256 of the gate‖up stack and its own down."""
        switch = self.switch_mlp
        slot = mx.full((*x.shape[:-1], 1), self.experts, dtype=mx.uint32)
        tokens = mx.expand_dims(x, (-2, -3))
        act = switch.activate(switch.gate_up_proj(tokens, slot, sorted_indices=True))
        return self.shared_expert.down_proj(act.squeeze(-2).squeeze(-2))

    def internals(self, x: mx.array) -> Qwen35MoEInternals:
        logits = self.gate(x)
        probs, chosen, weights = self.route(logits[..., : self.experts])
        routed = (self._routed(x, chosen) * mx.expand_dims(weights, -1)).sum(axis=-2)
        shared = mx.sigmoid(logits[..., self.experts :]) * self._shared(x)
        return Qwen35MoEInternals(probs, chosen, weights, routed, shared, routed + shared)

    def fused_step_applies(self) -> bool:
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        shared = self.shared_expert.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and isinstance(shared, nn.QuantizedLinear)
            # The gemv kernels read an affine bias per group; MXFP carries none.
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and (shared.bits, shared.group_size) == (down.bits, down.group_size)
            and moe_gemv_applies(
                self.hidden, self.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(self.experts, self.k)
        )

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array:
        """Four dispatches for the whole sparse block at T=1: the shared expert rides
        along as the ninth slot, so routing, silu, weighting, the expert sum and the
        residual all stay inside the two gemv kernels."""
        gate_up = self.switch_mlp.gate_up_proj
        down = self.switch_mlp.down_proj
        shared = self.shared_expert.down_proj
        assert isinstance(gate_up, QuantizedSwitchLinear)
        assert isinstance(down, QuantizedSwitchLinear)
        assert isinstance(shared, nn.QuantizedLinear)
        assert gate_up.biases is not None and down.biases is not None
        chosen, weights = softmax_topk(self.gate(x).reshape(-1), self.k, shared=True)
        act = moe_gate_up_act(
            x.reshape(-1), gate_up.weight, gate_up.scales, gate_up.biases, chosen,
            group_size=gate_up.group_size, bits=gate_up.bits,
        )
        return moe_down_combine(
            act.reshape(-1), down.weight, down.scales, down.biases, chosen, weights,
            residual.reshape(-1), group_size=down.group_size, bits=down.bits,
            shared=_packed(shared),
        ).reshape(1, 1, self.hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.internals(x).out
