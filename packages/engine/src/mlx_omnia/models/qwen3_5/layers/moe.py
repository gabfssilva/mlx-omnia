from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.down_combine import DownCombine
from mlx_omnia.core.kernels.gate_up import GateUp
from mlx_omnia.core.kernels.route import Route
from mlx_omnia.core.layers import (
    SORTED_GATHER_MIN,
    SegmentedLinear,
    SwitchLinear,
    sorted_gather,
)
from mlx_omnia.core.mxcompat import softmax
from mlx_omnia.models.qwen3_5.config import Qwen35TextConfig


class Qwen35SharedExpert(nn.Module):
    """The shared expert as its own three leaves, beside the routed stack rather than
    inside it — gate‖up in the stack's row-interleaved layout, so the same epilogue reads
    both. A stack holds one quantization format and a per-leaf plan gives this expert its
    own (the 35B oQ ships the routed experts at 3 bits and this one at 8); appending a
    257th row would also materialize the *down* tensor, the one that goes mmap'd from the
    file into the model (5.4 GB at 6 bits). The spare slot of the T=1 kernels is what
    keeps it inside the fused step regardless."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        inner = config.moe_intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, config.hidden_size, bias=False)


class Qwen35SwitchGLU(nn.Module):
    """gate‖up row-interleaved ([g0,u0,g1,…]) — one gather reads both projections of a
    routed expert."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
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
        self._route: Route | None = None
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

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
        """The shared expert through its own leaves, in the stack's interleaved layout."""
        shared = self.shared_expert
        return shared.down_proj(self.switch_mlp.activate(shared.gate_up_proj(x)))

    def internals(self, x: mx.array) -> Qwen35MoEInternals:
        logits = self.gate(x)
        probs, chosen, weights = self.route(logits[..., : self.experts])
        routed = (self._routed(x, chosen) * mx.expand_dims(weights, -1)).sum(axis=-2)
        shared = mx.sigmoid(logits[..., self.experts :]) * self._shared(x)
        return Qwen35MoEInternals(probs, chosen, weights, routed, shared, routed + shared)

    def _kernels(self) -> tuple[Route, GateUp, DownCombine]:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final."""
        route, gate_up, down = self._route, self._gate_up, self._down
        if route is None or gate_up is None or down is None:
            switch = self.switch_mlp
            gate = self.gate
            # The gate runs its own gemv below (its leaf may be quantized), so the
            # matrix here only ever serves a strategy that fuses one — and a router the
            # loader kept segmented has no single matrix to hand it.
            route = Route(
                None if isinstance(gate, SegmentedLinear) else gate.weight,
                experts=self.experts,
                k=self.k,
                shared=True,
            )
            gate_up = GateUp(
                switch.gate_up_proj,
                hidden=self.hidden,
                inner=switch.inner,
                shared=self.shared_expert.gate_up_proj,
            )
            down = DownCombine(
                switch.down_proj,
                hidden=self.hidden,
                inner=switch.inner,
                shared=self.shared_expert.down_proj,
            )
            self._route, self._gate_up, self._down = route, gate_up, down
        return route, gate_up, down

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array:
        """Four dispatches for the whole sparse block at T=1: the shared expert rides
        along as the ninth slot, so routing, silu, weighting, the expert sum and the
        residual all stay inside the two gemv kernels."""
        route, gate_up, down = self._kernels()
        chosen, weights = route(x.reshape(-1), logits=self.gate(x).reshape(-1))
        act = gate_up(x.reshape(-1), chosen)
        return down(act, chosen, weights, residual.reshape(-1)).reshape(1, 1, self.hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.internals(x).out
