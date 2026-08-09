import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import FixedKVCache, KVCache, RingKVCache
from sideros.core.kernels.dense_mlp import (
    dense_down_residual,
    dense_down_residual_applies,
    dense_gate_up_swiglu,
    dense_gate_up_swiglu_applies,
)
from sideros.core.kernels.residual_rms_router import (
    residual_rms_norm,
    residual_rms_norm_applies,
    residual_rms_router,
    residual_rms_router_applies,
)
from sideros.core.layers import SwiGLU
from sideros.models.laguna.config import LagunaConfig
from sideros.models.laguna.layers.attention import LagunaAttention
from sideros.models.laguna.layers.moe import LagunaSparseMoe


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

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | FixedKVCache | RingKVCache,
    ) -> mx.array:
        branch = self.self_attn(self.input_layernorm(x), mask, cache)
        mlp = self.mlp
        router_logits: mx.array | None = None
        router_keys: mx.array | None = None
        if (
            x.shape[1] == 1
            and isinstance(mlp, LagunaSparseMoe)
            and residual_rms_router_applies(x.shape[-1], mlp.split + mlp.k, 8)
        ):
            attended, h, router_logits, router_keys = residual_rms_router(
                x,
                branch,
                self.post_attention_layernorm.weight,
                mlp.gate.weight,
                mlp.e_score_correction_bias,
                eps=self.post_attention_layernorm.eps,
            )
        elif residual_rms_norm_applies(x.shape[-1]):
            attended, h = residual_rms_norm(
                x, branch, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps
            )
        else:
            attended = x + branch
            h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and isinstance(mlp, LagunaSparseMoe):
            if mlp.packed_step_applies():
                return mlp.packed_step(h, attended, router_logits, router_keys)
            if mlp.fused_step_applies():
                return mlp.fused_step(h, attended, router_logits)
        if x.shape[1] == 1 and isinstance(mlp, SwiGLU):
            fused = _dense_weights(mlp)
            if fused is not None:
                gate_up, down = fused
                row = h.reshape(-1)
                activated = dense_gate_up_swiglu(row, gate_up)
                return dense_down_residual(activated, down, attended.reshape(-1)).reshape(
                    attended.shape
                )
        return attended + self.mlp(h)


class LagunaTrunk(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LagunaBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def _dense_weights(mlp: SwiGLU) -> tuple[mx.array, mx.array] | None:
    """The fused dense pair, when both projections are the plain bf16 arrays the kernels
    read. A quantized dense MLP has no dense weight to hand them and stays on the ops."""
    gate_up = mlp.gate_up_proj.weight
    down = mlp.down_proj.weight
    if gate_up.dtype != mx.bfloat16 or down.dtype != mx.bfloat16:
        return None
    inner, hidden = down.shape[1], down.shape[0]
    if not dense_gate_up_swiglu_applies(hidden, inner, gate_up.dtype):
        return None
    if not dense_down_residual_applies(hidden, inner, gate_up.dtype):
        return None
    return gate_up, down
