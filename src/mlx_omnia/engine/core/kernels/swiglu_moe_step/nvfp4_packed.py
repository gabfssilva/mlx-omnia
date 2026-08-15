"""The packed NVFP4 arrangement of the gated T=1 step.

Routing hands the gate/up kernel the router's ordinal keys, so expert selection happens
inside that dispatch. The shared expert — quantized on its own schedule, so barred from
the routed stack's slots — runs the halved-scale gate/up kernel and rides the down
kernel's unrouted slot, and the routed scaling folds into the gains. The arrangement
stands only when every piece resolves to its kernel: each sub-build declines on its own
terms and the decline propagates.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.down_combine import Nvfp4PackedDownCombine
from mlx_omnia.engine.core.kernels.gate_up import Nvfp4PackedGateUp, OrdinalRouting
from mlx_omnia.engine.core.kernels.mlp.nvfp4 import (
    halve_gate_up_scales,
    nvfp4_halved_gate_up,
    nvfp4_halved_gate_up_applies,
)
from mlx_omnia.engine.core.kernels.route.kernel import Routing, gate_logits
from mlx_omnia.engine.core.kernels.route.ordinal import (
    OrdinalRoute,
    ordinal_keys,
    router_tournament,
)
from mlx_omnia.engine.core.kernels.swiglu_moe_step.kernel import SwigluMoeStepStrategy
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, SwiGLU, SwitchLinear


@dataclass(frozen=True)
class Nvfp4PackedSwigluMoeStep(SwigluMoeStepStrategy):
    route: OrdinalRoute
    gate_up: Nvfp4PackedGateUp
    down: Nvfp4PackedDownCombine
    gate: mx.array
    bias: mx.array
    shared_weight: mx.array
    shared_scales: mx.array
    scale: float
    k: int

    @classmethod
    def build(
        cls,
        *,
        gate: mx.array,
        bias: mx.array,
        experts: int,
        k: int,
        scale: float,
        softcap: float,
        gate_up_proj: SwitchLinear | QuantizedSwitchLinear,
        down_proj: SwitchLinear | QuantizedSwitchLinear,
        hidden: int,
        inner: int,
        shared: SwiGLU,
    ) -> Self | None:
        # The keys never see the softcap, so a capping declaration cannot pack.
        if softcap != 0.0:
            return None
        shared_gate_up = shared.gate_up_proj
        if not isinstance(shared_gate_up, nn.QuantizedLinear):
            return None
        if shared_gate_up.mode != "nvfp4" or not nvfp4_halved_gate_up_applies(hidden, shared.inner):
            return None
        route = OrdinalRoute.build(
            gate, routing=Routing(experts=experts, k=k, scoring="sigmoid", bias=bias)
        )
        if route is None:
            return None
        gate_up = Nvfp4PackedGateUp.build(
            gate_up_proj,
            hidden=hidden,
            inner=inner,
            activation="silu",
            limit=None,
            bias=None,
            layout="interleaved",
            routing=OrdinalRouting(k),
            shared=None,
        )
        if gate_up is None:
            return None
        down = Nvfp4PackedDownCombine.build(
            down_proj,
            hidden=hidden,
            inner=inner,
            bias=None,
            shared=shared.down_proj,
            layout="interleaved",
        )
        if down is None:
            return None
        scales = halve_gate_up_scales(shared_gate_up.scales)
        if scales is None:
            return None
        mx.eval(scales)
        return cls(
            route, gate_up, down, gate, bias, shared_gate_up.weight, scales, scale, k
        )

    @property
    def worthwhile(self) -> bool:
        return True

    def __call__(
        self,
        x: mx.array,
        residual: mx.array,
        *,
        logits: mx.array | None = None,
        keys: mx.array | None = None,
    ) -> mx.array:
        row = x.reshape(-1)
        row_logits = gate_logits(self.gate, row, logits).astype(mx.float32)
        row_keys = (
            ordinal_keys(mx.sigmoid(row_logits) + self.bias)
            if keys is None
            else keys.reshape(-1)
        )
        chosen, weights = self.route(row, logits=row_logits)
        routed = self.gate_up(row, row_keys)
        shared = nvfp4_halved_gate_up(row, self.shared_weight, self.shared_scales)
        # The unrouted slot is the last row of the pack: the down kernel reads its
        # activation there and never reads its index or its weight.
        act = mx.concatenate([routed, shared.reshape(1, -1)], axis=0)
        slots = mx.concatenate([chosen.astype(mx.uint32), _SPARE_SLOT])
        gains = mx.concatenate(
            [weights * self.scale, mx.ones((1,), dtype=weights.dtype)]
        )
        return self.down(act, slots, gains, residual.reshape(-1)).reshape(x.shape)

    def batch(self, h: mx.array, residual: mx.array) -> mx.array | None:
        logits = mx.matmul(h, self.gate.T).astype(mx.float32)
        chosen, weights = router_tournament(logits, self.bias, self.k)
        keys = ordinal_keys(mx.sigmoid(logits) + self.bias)
        routed = self.gate_up(h, keys)
        shared = nvfp4_halved_gate_up(h, self.shared_weight, self.shared_scales)
        act = mx.concatenate([routed, shared[..., None, :]], axis=-2)
        spare = mx.zeros((*chosen.shape[:-1], 1), dtype=mx.uint32)
        slots = mx.concatenate([chosen.astype(mx.uint32), spare], axis=-1)
        gains = mx.concatenate(
            [
                weights * self.scale,
                mx.ones((*weights.shape[:-1], 1), dtype=weights.dtype),
            ],
            axis=-1,
        )
        return self.down(act, slots, gains, residual)


_SPARE_SLOT = mx.zeros((1,), dtype=mx.uint32)
# Evaluated at import: a pending node built on the importing thread carries that
# thread's stream, and evaluating it from a worker thread (the server's engine) has
# no encoder for it — "There is no Stream(gpu, 0) in current thread".
mx.eval(_SPARE_SLOT)
