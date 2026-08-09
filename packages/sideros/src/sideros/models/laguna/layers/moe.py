import mlx.core as mx
import mlx.nn as nn

from sideros.core.kernels.down_combine import DownCombine
from sideros.core.kernels.gate_up import GateUp
from sideros.core.kernels.moe_route import sigmoid_topk, softmax_topk_applies
from sideros.core.kernels.nvfp4_fused_down_residual import (
    halve_down_scales,
    nvfp4_fused_down_residual,
    nvfp4_fused_down_residual_applies,
)
from sideros.core.kernels.nvfp4_halved_gate_up import (
    halve_gate_up_scales,
    nvfp4_halved_gate_up,
)
from sideros.core.kernels.nvfp4_packed_gate_up import (
    nvfp4_packed_gate_up,
    nvfp4_packed_gate_up_applies,
    pack_gate_up_scales,
)
from sideros.core.kernels.router_ordinal import router_tournament
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
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

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

    def packed_step_applies(self) -> bool:
        """The packed NVFP4 path: one dispatch for the routed gate/up over the top-k the
        ordinal keys pick, one for the shared stack, one for both down-projections joined
        with the residual."""
        inner = self.switch_mlp.inner
        return (
            nvfp4_packed_gate_up_applies(self.hidden, inner, self.split + self.k, self.k)
            and nvfp4_fused_down_residual_applies(self.hidden, inner, self.k)
            and _packed_banks(self) is not None
        )

    def packed_step(
        self,
        h: mx.array,
        residual: mx.array,
        router_logits: mx.array | None = None,
        router_keys: mx.array | None = None,
    ) -> mx.array:
        banks = _packed_banks(self)
        assert banks is not None
        relaid_weight, packed_scales, down_halved, shared_halved, shared_down_scales = banks
        down = self.switch_mlp.down_proj
        row = h.reshape(-1)

        logits = (
            self.gate(h).reshape(-1).astype(mx.float32)
            if router_logits is None
            else router_logits.reshape(-1).astype(mx.float32)
        )
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias
        keys = _ordinal_keys(biased) if router_keys is None else router_keys.reshape(-1)
        chosen, weights = router_tournament(logits, self.e_score_correction_bias, self.k)

        routed = nvfp4_packed_gate_up(row, relaid_weight, packed_scales, keys, self.k)
        shared = nvfp4_halved_gate_up(
            row, self.shared_expert.gate_up_proj.weight, shared_halved
        )
        return nvfp4_fused_down_residual(
            routed,
            down.weight,
            down_halved,
            chosen.astype(mx.uint32),
            weights,
            shared,
            self.shared_expert.down_proj.weight,
            shared_down_scales,
            residual.reshape(-1),
            self.scaling,
        ).reshape(1, 1, self.hidden)

    def _kernels(self) -> tuple[GateUp, DownCombine]:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final."""
        gate_up, down = self._gate_up, self._down
        if gate_up is None or down is None:
            switch = self.switch_mlp
            gate_up = GateUp(switch.gate_up_proj, hidden=self.hidden, inner=switch.inner)
            down = DownCombine(switch.down_proj, hidden=self.hidden, inner=switch.inner)
            self._gate_up, self._down = gate_up, down
        return gate_up, down

    def fused_step_applies(self) -> bool:
        # The routing kernel skips the softcap; a capped config takes the op chain.
        return self.cap == 0.0 and softmax_topk_applies(self.split + self.k, self.k)

    def fused_step(
        self, h: mx.array, residual: mx.array, router_logits: mx.array | None = None
    ) -> mx.array:
        gate_up, down = self._kernels()
        chosen, weights = sigmoid_topk(
            self.gate(h).reshape(-1) if router_logits is None else router_logits.reshape(-1),
            self.e_score_correction_bias,
            self.k,
            scale=self.scaling,
        )
        # The shared expert quantizes at different bits than the routed stack, so
        # it can't ride the kernel's shared slot; it folds into the residual input
        # instead, and routed_scaling folds into the routing weights.
        combined = residual + self.shared_expert(h)
        act = gate_up(h.reshape(-1), chosen)
        return down(act, chosen, weights, combined.reshape(-1)).reshape(1, 1, self.hidden)

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


def _ordinal_keys(scores: mx.array) -> mx.array:
    """`router_ordinal`'s remapping, in ops: ascending unsigned order over the keys is
    descending routing score. The kernel derives it from `-(sigmoid + bias)`, so the
    negation happens here and the bit transform is the same monotone IEEE one."""
    bits = mx.view(-scores.astype(mx.float32), mx.uint32)
    negative = (bits & 0x80000000) != 0
    return mx.where(negative, ~bits, bits ^ 0x80000000).astype(mx.uint32)


# Off the module for the same reason as the rope atlas: an mx.array assigned to an
# nn.Module attribute joins the parameter tree, and these are derived layouts.
_BANKS: dict[int, tuple[mx.array, ...] | tuple[()]] = {}


def _their_row_order(rows: int) -> mx.array:
    """Row permutation from this engine's fused bank to the one the packed kernel walks.

    `SwitchGLU` stacks gate and up interleaved per row (`[g0, u0, g1, u1, ...]`); the kernel
    addresses tiles of 32 (`gate_row = (row / 32) * 64 + row % 32`, up at `+32`) and encodes
    that in both its weight reads and its scale walk, so the bank is reordered once at load
    rather than the kernel bent to fit.
    """
    logical = rows // 2
    order = [0] * rows
    for row in range(logical):
        gate_row = (row // 32) * 64 + row % 32
        order[gate_row] = row * 2
        order[gate_row + 32] = row * 2 + 1
    return mx.array(order, dtype=mx.int32)


def _packed_banks(moe: "LagunaSparseMoe") -> tuple[mx.array, ...] | None:
    """The four load-time side layouts the packed step reads, or `None` when the
    checkpoint fails any of their certificates. Built once and kept on the module."""
    cached = _BANKS.get(id(moe))
    if cached is not None:
        return cached if cached != () else None
    gate_up = moe.switch_mlp.gate_up_proj
    down = moe.switch_mlp.down_proj
    shared_gate_up = moe.shared_expert.gate_up_proj
    shared_down = moe.shared_expert.down_proj
    built: tuple[mx.array, ...] | None = None
    if (
        isinstance(gate_up, QuantizedSwitchLinear)
        and isinstance(down, QuantizedSwitchLinear)
        and gate_up.mode == "nvfp4"
        and down.mode == "nvfp4"
    ):
        order = _their_row_order(gate_up.scales.shape[1])
        relaid_weight = mx.contiguous(mx.take(gate_up.weight, order, axis=1))
        relaid_scales = mx.take(gate_up.scales, order, axis=1)
        packed = pack_gate_up_scales(relaid_scales)
        down_halved = halve_down_scales(down.scales)
        shared_halved = halve_gate_up_scales(shared_gate_up.scales)
        shared_down_halved = halve_down_scales(shared_down.scales)
        if (
            packed is not None
            and down_halved is not None
            and shared_halved is not None
            and shared_down_halved is not None
        ):
            mx.eval(relaid_weight, packed, down_halved, shared_halved, shared_down_halved)
            built = (
                relaid_weight,
                packed,
                down_halved,
                shared_halved,
                shared_down_halved,
            )
    _BANKS[id(moe)] = built if built is not None else ()
    return built
