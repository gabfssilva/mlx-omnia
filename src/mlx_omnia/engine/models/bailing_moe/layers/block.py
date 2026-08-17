import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.bailing_moe.config import BailingMoEConfig
from mlx_omnia.engine.models.bailing_moe.layers.attention import BailingMoEAttention
from mlx_omnia.engine.models.bailing_moe.layers.moe import BailingMoEMLP


class BailingMoEBlock(nn.Module):
    def __init__(self, config: BailingMoEConfig, routes: bool) -> None:
        super().__init__()
        self.attention = BailingMoEAttention(config)
        self.mlp: BailingMoEMLP | SwiGLU = (
            BailingMoEMLP(config)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        attended = x + self.attention(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))
