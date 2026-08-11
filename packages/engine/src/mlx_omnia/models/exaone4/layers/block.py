import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.core.layers import SwiGLU
from mlx_omnia.models.exaone4.config import Exaone4Config
from mlx_omnia.models.exaone4.layers.attention import Exaone4Attention


class Exaone4Block(nn.Module):
    def __init__(self, config: Exaone4Config, local: bool) -> None:
        super().__init__()
        self.self_attn = Exaone4Attention(config, local)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(x, cache))
        return attended + self.post_feedforward_layernorm(self.mlp(attended))
