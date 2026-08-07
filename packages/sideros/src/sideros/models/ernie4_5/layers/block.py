import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU
from sideros.models.ernie4_5.config import Ernie45Config
from sideros.models.ernie4_5.layers.attention import Ernie45Attention


class Ernie45Block(nn.Module):
    def __init__(self, config: Ernie45Config) -> None:
        super().__init__()
        self.self_attn = Ernie45Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))
