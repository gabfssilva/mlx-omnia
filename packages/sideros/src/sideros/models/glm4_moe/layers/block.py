import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU
from sideros.models.glm4_moe.config import Glm4MoEConfig
from sideros.models.glm4_moe.layers.attention import Glm4MoEAttention
from sideros.models.glm4_moe.layers.moe import Glm4MoEMLP


class Glm4MoEBlock(nn.Module):
    def __init__(self, config: Glm4MoEConfig, routes: bool) -> None:
        super().__init__()
        self.self_attn = Glm4MoEAttention(config)
        self.mlp: Glm4MoEMLP | SwiGLU = (
            Glm4MoEMLP(config)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Glm4MoETrunk(nn.Module):
    def __init__(self, config: Glm4MoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm4MoEBlock(config, layer >= config.first_k_dense_replace)
            for layer in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
