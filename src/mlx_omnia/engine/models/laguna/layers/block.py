from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.add_norm import AddRmsNorm
from mlx_omnia.engine.core.kernels.attention import AttentionCache
from mlx_omnia.engine.core.kernels.mlp import Mlp
from mlx_omnia.engine.core.kernels.route.residual import (
    residual_rms_router,
    residual_rms_router_applies,
)
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.laguna.config import LagunaConfig
from mlx_omnia.engine.models.laguna.layers.attention import LagunaAttention
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe


class _Kernels(NamedTuple):
    """The residual join and, when the block is dense, its MLP — resolved once.

    `router` is the router matrix and its correction bias when the fused join can also
    carry the gate's gemv, `None` otherwise. That fusion has no delegator of its own: it
    produces two extra outputs the residual primitive does not name, so it stays a
    module-path dispatch whose predicate is answered here rather than per step.
    """

    add_norm: AddRmsNorm
    mlp: Mlp | None
    router: tuple[mx.array, mx.array] | None


class LagunaBlock(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LagunaAttention(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = LagunaSparseMoe(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._hidden = config.hidden_size
        self._inner = config.intermediate_size
        self._resolved: _Kernels | None = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: AttentionCache,
    ) -> mx.array:
        branch = self.self_attn(self.input_layernorm(x), mask, cache)
        kernels = self._kernels()
        mlp = self.mlp
        step = x.shape[1] == 1
        router_logits: mx.array | None = None
        router_keys: mx.array | None = None
        if step and kernels.router is not None:
            router_weight, correction = kernels.router
            attended, h, router_logits, router_keys = residual_rms_router(
                x,
                branch,
                self.post_attention_layernorm.weight,
                router_weight,
                correction,
                eps=self.post_attention_layernorm.eps,
            )
        else:
            attended, h = kernels.add_norm(x, branch)
        if step and isinstance(mlp, LagunaSparseMoe) and mlp.step_applies():
            return mlp.step(h, attended, router_logits, router_keys)
        if step and kernels.mlp is not None:
            return kernels.mlp(h.reshape(-1), attended.reshape(-1)).reshape(attended.shape)
        return attended + mlp(h)

    def _kernels(self) -> _Kernels:
        """Resolved once, at the first step — after load, when the leaves' formats are
        final."""
        kernels = self._resolved
        if kernels is None:
            mlp = self.mlp
            router = (
                (mlp.gate.weight, mlp.e_score_correction_bias)
                if isinstance(mlp, LagunaSparseMoe)
                and residual_rms_router_applies(self._hidden, mlp.experts, 8)
                else None
            )
            kernels = _Kernels(
                AddRmsNorm(self.post_attention_layernorm),
                Mlp(mlp, hidden=self._hidden, inner=self._inner)
                if isinstance(mlp, SwiGLU)
                else None,
                router,
            )
            self._resolved = kernels
        return kernels


class LagunaTrunk(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LagunaBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
