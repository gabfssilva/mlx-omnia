import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.olmo2.config import Olmo2Config
from mlx_omnia.engine.models.olmo2.layers.attention import Olmo2Attention


class Olmo2Block(nn.Module):
    def __init__(self, config: Olmo2Config) -> None:
        super().__init__()
        self.self_attn = Olmo2Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(x, cache))
        return attended + self.post_feedforward_layernorm(self.mlp(attended))
