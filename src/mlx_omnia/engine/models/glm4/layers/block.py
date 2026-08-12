import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.glm4.config import Glm4Config
from mlx_omnia.engine.models.glm4.layers.attention import Glm4Attention


class Glm4Block(nn.Module):
    def __init__(self, config: Glm4Config) -> None:
        super().__init__()
        self.self_attn = Glm4Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.post_self_attn_layernorm(
            self.self_attn(self.input_layernorm(x), cache)
        )
        return attended + self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(attended)))
