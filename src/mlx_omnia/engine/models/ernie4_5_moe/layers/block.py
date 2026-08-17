import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.ernie4_5_moe.config import Ernie45MoEConfig
from mlx_omnia.engine.models.ernie4_5_moe.layers.attention import Ernie45MoEAttention
from mlx_omnia.engine.models.ernie4_5_moe.layers.moe import Ernie45MoEMLP


class Ernie45MoEBlock(nn.Module):
    def __init__(self, config: Ernie45MoEConfig, routes: bool) -> None:
        super().__init__()
        self.self_attn = Ernie45MoEAttention(config)
        self.mlp: Ernie45MoEMLP | SwiGLU = (
            Ernie45MoEMLP(config)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))
