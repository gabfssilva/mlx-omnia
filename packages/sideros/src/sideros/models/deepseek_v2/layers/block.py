import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU
from sideros.core.rope import Yarn
from sideros.models.deepseek_v2.config import DeepseekV2Config
from sideros.models.deepseek_v2.layers.attention import DeepseekV2Attention
from sideros.models.deepseek_v2.layers.moe import DeepseekV2MoE


class DeepseekV2Block(nn.Module):
    def __init__(self, config: DeepseekV2Config, rope: Yarn, routes: bool) -> None:
        super().__init__()
        self.self_attn = DeepseekV2Attention(config, rope)
        self.mlp: DeepseekV2MoE | SwiGLU = (
            DeepseekV2MoE(config)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))
