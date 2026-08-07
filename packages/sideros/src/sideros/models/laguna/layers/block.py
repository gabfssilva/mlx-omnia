import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
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
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(attended)
        mlp = self.mlp
        if x.shape[1] == 1 and isinstance(mlp, LagunaSparseMoe) and mlp.fused_step_applies():
            return mlp.fused_step(h, attended)
        return attended + self.mlp(h)


class LagunaTrunk(nn.Module):
    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LagunaBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
