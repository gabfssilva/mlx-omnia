import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.core.layers import SwiGLU
from mlx_omnia.models.granite.config import GraniteConfig
from mlx_omnia.models.granite.layers.attention import GraniteAttention


class GraniteBlock(nn.Module):
    def __init__(self, config: GraniteConfig) -> None:
        super().__init__()
        self.residual_multiplier = config.residual_multiplier
        self.self_attn = GraniteAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache) * self.residual_multiplier
        mixed = self.mlp(self.post_attention_layernorm(attended))
        return attended + mixed * self.residual_multiplier
