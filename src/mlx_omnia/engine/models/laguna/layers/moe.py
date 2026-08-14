import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.swiglu_moe_step import SwigluMoeStep
from mlx_omnia.engine.core.layers import SORTED_GATHER_MIN, SwiGLU, SwitchGLU, sorted_gather
from mlx_omnia.engine.models.laguna.config import LagunaConfig


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
        self._resolved: SwigluMoeStep | None = None

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

    def _step(self) -> SwigluMoeStep:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final."""
        step = self._resolved
        if step is None:
            switch = self.switch_mlp
            step = SwigluMoeStep(
                gate=self.gate.weight,
                bias=self.e_score_correction_bias,
                experts=self.experts,
                k=self.k,
                scale=self.scaling,
                softcap=self.cap,
                gate_up_proj=switch.gate_up_proj,
                down_proj=switch.down_proj,
                hidden=self.hidden,
                inner=switch.inner,
                shared=self.shared_expert,
            )
            self._resolved = step
        return step

    def step_applies(self) -> bool:
        return self._step().worthwhile

    def step(
        self,
        h: mx.array,
        residual: mx.array,
        router_logits: mx.array | None = None,
        router_keys: mx.array | None = None,
    ) -> mx.array:
        return self._step()(h, residual, logits=router_logits, keys=router_keys)

    def batch_step(self, h: mx.array, residual: mx.array) -> mx.array:
        batched = self._step().batch(h, residual)
        return residual + self(h) if batched is None else batched

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.route(x)
        tokens = x.size // self.hidden
        if tokens * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        routed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        return routed * self.scaling + self.shared_expert(x)
