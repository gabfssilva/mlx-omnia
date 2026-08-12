import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.moe_step import MoeStep
from mlx_omnia.engine.core.kernels.route import Route
from mlx_omnia.engine.core.layers import SORTED_GATHER_MIN, sorted_gather
from mlx_omnia.engine.models.nemotron_h.config import NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.layers.mlp import NemotronHMLP, SwitchMLP


class NemotronHGate(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.routed_experts,))
        self.k = config.experts_per_tok
        self.n_group = config.expert_groups
        self.topk_group = config.expert_groups_kept
        self.scaling = config.routed_scale
        self.norm_topk = config.normalize_topk
        self._route: Route | None = None

    def _decode_route(self) -> Route:
        """Resolved once, at the first step — after load, when the weights are final."""
        route = self._route
        if route is None:
            route = Route(
                self.weight,
                experts=self.weight.shape[0],
                k=self.k,
                scoring="sigmoid",
                bias=self.e_score_correction_bias,
                normalize=self.norm_topk,
                norm_eps=1e-20,
                scale=self.scaling,
            )
            self._route = route
        return route

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        length = x.shape[-2]
        if length <= 4 and self.n_group == 1:
            # The logits come from one gemv per row — a batched dense gemm is not
            # row-for-row identical to the gemv (bf16 reduction order), and a verify
            # that rounds differently from the decode it stands in for flips
            # near-ties — and the pick itself rides one dispatch for all rows.
            from mlx_omnia.engine.core.kernels.route import SigmoidWideRoute

            route = self._decode_route()
            strategy = route.strategy
            # Measured dead end: fusing the gemv into the pick dispatch (a bit-exact
            # `gemv_al_bfloat16` replica into threadgroup memory) was timing-neutral in
            # both compiled decode and compiled verify — these dispatches already
            # overlap; the launches are not on the critical path.
            if length > 1 and isinstance(strategy, SigmoidWideRoute):
                logits = mx.stack([row @ self.weight.T for row in x[0]])
                chosen, weights = strategy.rows(logits)
                return chosen[None], weights[None]
            picks = [route(row, logits=(row @ self.weight.T)) for row in x[0]]
            chosen = mx.stack([pick[0] for pick in picks])[None]
            weights = mx.stack([pick[1] for pick in picks])[None]
            return chosen, weights
        scores = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        selector = scores + self.e_score_correction_bias
        if self.n_group > 1:
            grouped = mx.unflatten(selector, axis=-1, shape=(self.n_group, -1))
            strength = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
            dropped = self.n_group - self.topk_group
            worst = mx.argpartition(strength, kth=dropped - 1, axis=-2)[..., :dropped, :]
            selector = mx.flatten(
                mx.put_along_axis(grouped, worst, mx.array(0.0), axis=-2), -2, -1
            )
        chosen = mx.argpartition(-selector, kth=self.k - 1, axis=-1)[..., : self.k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.k > 1 and self.norm_topk:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.scaling


class NemotronHMoE(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        latent = config.moe_latent_size
        self.gate = NemotronHGate(config)
        self.switch_mlp = SwitchMLP(
            config.routed_experts,
            latent if latent is not None else config.hidden_size,
            config.moe_inner,
        )
        if config.shared_expert_inner:
            self.shared_experts = NemotronHMLP(
                config.hidden_size, config.shared_expert_inner, config.mlp_bias
            )
        if latent is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, latent, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(latent, config.hidden_size, bias=config.mlp_bias)
        self.inner = latent if latent is not None else config.hidden_size
        self.k = config.experts_per_tok
        self._step: MoeStep | None = None

    def _decode_step(self) -> MoeStep:
        """Resolved once, at the first step — after load, when the weights are final."""
        step = self._step
        if step is None:
            step = MoeStep(
                fc1=self.switch_mlp.fc1,
                fc2=self.switch_mlp.fc2,
                hidden=self.inner,
                # The out-rows axis is never packed, so fc1's row count is the expert
                # width whatever the format.
                inner=self.switch_mlp.fc1.weight.shape[1],
                # Measured dead end: folding the shared expert into the two dispatches
                # (`shared=`) lost ~0.1 ms/token — its down projection serializes after
                # the routed experts, where the separate ops overlapped freely.
            )
            self._step = step
        return step

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.gate(x)
        projected = self.fc1_latent_proj(x) if "fc1_latent_proj" in self else x
        length = projected.shape[-2]
        if length == 1 and "fc2_latent_proj" not in self:
            step = self._decode_step()
            routed_row = step(projected[0, 0], chosen[0, 0], weights[0, 0])
            if "shared_experts" in self:
                routed_row = routed_row + self.shared_experts(x)[0, 0]
            return routed_row[None, None]
        if 1 < length <= 4 and "fc2_latent_proj" not in self:
            from mlx_omnia.engine.core.kernels.moe_step import Nvfp4MoeStep

            strategy = self._decode_step().strategy
            if isinstance(strategy, Nvfp4MoeStep):
                mixed = strategy.rows(projected[0], chosen[0], weights[0])[None]
                if "shared_experts" in self:
                    return mixed + self.shared_experts(x)
                return mixed
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(projected, chosen, k=self.k, hidden=self.inner, apply=apply)
        else:
            tokens = mx.expand_dims(projected, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        mixed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2).astype(x.dtype)
        if "fc2_latent_proj" in self:
            mixed = self.fc2_latent_proj(mixed)
        if "shared_experts" in self:
            return mixed + self.shared_experts(x)
        return mixed
