from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.down_combine import DownCombine, Nvfp4PackedDownCombine
from mlx_omnia.core.kernels.gate_up import GateUp, Nvfp4PackedGateUp, OrdinalRouting
from mlx_omnia.core.kernels.mlp.nvfp4 import halve_gate_up_scales, nvfp4_halved_gate_up
from mlx_omnia.core.kernels.route import DefaultRoute, OrdinalRoute, Route
from mlx_omnia.core.layers import SORTED_GATHER_MIN, SwiGLU, SwitchGLU, sorted_gather
from mlx_omnia.models.laguna.config import LagunaConfig


class _Kernels(NamedTuple):
    """One resolution of the whole T=1 sparse block.

    Two declarations of the same layer, and the resolution picks between them: the
    packed one hands routing to the gate/up kernel and the shared expert to the
    down/combine kernel's unrouted slot, and only stands when every piece of it
    resolved to its kernel; otherwise the layer is declared without either, which the
    delegators serve down to the op chain.
    """

    router: Route
    gate_up: GateUp
    down: DownCombine
    shared_scales: mx.array | None
    packed: bool


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
        self.experts = config.num_experts
        self.k = config.num_experts_per_tok
        self.split = config.num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.moe_routed_scaling_factor
        self.cap = config.moe_router_logit_softcapping
        self._resolved: _Kernels | None = None

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

    def _kernels(self) -> _Kernels:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final."""
        kernels = self._resolved
        if kernels is not None:
            return kernels
        switch = self.switch_mlp
        shared = self.shared_expert
        router = Route(
            self.gate.weight,
            experts=self.experts,
            k=self.k,
            scoring="sigmoid",
            bias=self.e_score_correction_bias,
        )
        gate_up = GateUp(
            switch.gate_up_proj,
            hidden=self.hidden,
            inner=switch.inner,
            routing=OrdinalRouting(self.k),
        )
        down = DownCombine(
            switch.down_proj,
            hidden=self.hidden,
            inner=switch.inner,
            shared=shared.down_proj,
        )
        scales = (
            halve_gate_up_scales(shared.gate_up_proj.scales)
            if isinstance(shared.gate_up_proj, nn.QuantizedLinear)
            else None
        )
        if (
            isinstance(router.strategy, OrdinalRoute)
            and isinstance(gate_up.strategy, Nvfp4PackedGateUp)
            and isinstance(down.strategy, Nvfp4PackedDownCombine)
            and scales is not None
        ):
            mx.eval(scales)
            kernels = _Kernels(router, gate_up, down, scales, True)
        else:
            kernels = _Kernels(
                Route(
                    self.gate.weight,
                    experts=self.experts,
                    k=self.k,
                    scoring="sigmoid",
                    bias=self.e_score_correction_bias,
                    scale=self.scaling,
                    softcap=self.cap,
                ),
                GateUp(switch.gate_up_proj, hidden=self.hidden, inner=switch.inner),
                DownCombine(switch.down_proj, hidden=self.hidden, inner=switch.inner),
                None,
                False,
            )
        self._resolved = kernels
        return kernels

    def step_applies(self) -> bool:
        """The op path stays for the declaration no kernel serves: the delegators are
        total, so what is left to ask is whether anything but the default answered."""
        kernels = self._kernels()
        return kernels.packed or not isinstance(kernels.router.strategy, DefaultRoute)

    def step(
        self,
        h: mx.array,
        residual: mx.array,
        router_logits: mx.array | None = None,
        router_keys: mx.array | None = None,
    ) -> mx.array:
        kernels = self._kernels()
        row = h.reshape(-1)
        if not kernels.packed:
            chosen, weights = kernels.router(
                row, logits=None if router_logits is None else router_logits.reshape(-1)
            )
            # The shared expert quantizes at different bits than the routed stack, so
            # it can't ride the kernel's shared slot; it folds into the residual input
            # instead, and routed_scaling folds into the routing weights.
            combined = residual + self.shared_expert(h)
            act = kernels.gate_up(row, chosen)
            return kernels.down(act, chosen, weights, combined.reshape(-1)).reshape(
                1, 1, self.hidden
            )

        shared_scales = kernels.shared_scales
        assert shared_scales is not None
        logits = (
            self.gate(h).reshape(-1).astype(mx.float32)
            if router_logits is None
            else router_logits.reshape(-1).astype(mx.float32)
        )
        keys = (
            _ordinal_keys(mx.sigmoid(logits) + self.e_score_correction_bias)
            if router_keys is None
            else router_keys.reshape(-1)
        )
        chosen, weights = kernels.router(row, logits=logits)
        routed = kernels.gate_up(row, keys)
        shared = nvfp4_halved_gate_up(
            row, self.shared_expert.gate_up_proj.weight, shared_scales
        )
        # The unrouted slot is the last row of the pack: the down kernel reads its
        # activation there and never reads its index or its weight.
        act = mx.concatenate([routed, shared.reshape(1, -1)], axis=0)
        slots = mx.concatenate([chosen.astype(mx.uint32), _SPARE_SLOT])
        gains = mx.concatenate(
            [weights * self.scaling, mx.ones((1,), dtype=weights.dtype)]
        )
        return kernels.down(act, slots, gains, residual.reshape(-1)).reshape(1, 1, self.hidden)

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


_SPARE_SLOT = mx.zeros((1,), dtype=mx.uint32)
# Evaluated at import: a pending node built on the importing thread carries that
# thread's stream, and evaluating it from a worker thread (the server's engine) has
# no encoder for it — "There is no Stream(gpu, 0) in current thread".
mx.eval(_SPARE_SLOT)


def _ordinal_keys(scores: mx.array) -> mx.array:
    """`OrdinalRoute`'s remapping, in ops: ascending unsigned order over the keys is
    descending routing score. The kernel derives it from `-(sigmoid + bias)`, so the
    negation happens here and the bit transform is the same monotone IEEE one."""
    bits = mx.view(-scores.astype(mx.float32), mx.uint32)
    negative = (bits & 0x80000000) != 0
    return mx.where(negative, ~bits, bits ^ 0x80000000).astype(mx.uint32)
