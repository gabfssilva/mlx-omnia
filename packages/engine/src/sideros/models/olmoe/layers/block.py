import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.olmoe.config import OlmoEConfig
from sideros.models.olmoe.layers.attention import OlmoEAttention
from sideros.models.olmoe.layers.moe import OlmoEMLP


class OlmoEBlock(nn.Module):
    def __init__(self, config: OlmoEConfig) -> None:
        super().__init__()
        self.self_attn = OlmoEAttention(config)
        self.mlp = OlmoEMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))
