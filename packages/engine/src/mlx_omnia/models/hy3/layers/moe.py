import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.down_combine import DownCombine
from mlx_omnia.core.kernels.gate_up import GateUp
from mlx_omnia.core.kernels.route import Route
from mlx_omnia.core.layers import SORTED_GATHER_MIN, SwiGLU, SwitchLinear, sorted_gather
from mlx_omnia.models.hy3.config import Hy3Config


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
        self.experts = config.num_experts
        self.k = config.num_experts_per_tok
        self.split = self.experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.router_scaling_factor
        self.fp32_combine = config.enable_moe_fp32_combine
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None
        self._route: Route | None = None

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

    def _kernels(self) -> tuple[GateUp, DownCombine, Route]:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final."""
        gate_up, down, route = self._gate_up, self._down, self._route
        if gate_up is None or down is None or route is None:
            switch = self.switch_mlp
            gate_up = GateUp(switch.gate_up_proj, hidden=self.hidden, inner=switch.inner)
            down = DownCombine(switch.down_proj, hidden=self.hidden, inner=switch.inner)
            route = Route(
                self.gate.weight,
                experts=self.experts,
                k=self.k,
                scoring="sigmoid",
                bias=self.e_score_correction_bias,
                scale=self.scaling,
            )
            self._gate_up, self._down, self._route = gate_up, down, route
        return gate_up, down, route

    def fused_step_applies(self) -> bool:
        return not self.fp32_combine

    def fused_step(self, x: mx.array, residual: mx.array) -> mx.array:
        gate_up, down, route = self._kernels()
        logits = self.gate(x.astype(mx.float32)).astype(x.dtype).reshape(-1)
        chosen, weights = route(x.reshape(-1), logits=logits)
        act = gate_up(x.reshape(-1), chosen)
        joined = residual + self.shared_expert(x)
        return down(act, chosen, weights, joined.reshape(-1)).reshape(1, 1, self.hidden)

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
