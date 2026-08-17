import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.core.masks import SLIDING
from mlx_omnia.engine.models.afmoe.config import AfmoeConfig
from mlx_omnia.engine.models.afmoe.layers.attention import AfmoeAttention
from mlx_omnia.engine.models.afmoe.layers.moe import AfmoeMLP


class AfmoeBlock(nn.Module):
    def __init__(self, config: AfmoeConfig, layer: int, layer_type: str) -> None:
        super().__init__()
        self.self_attn = AfmoeAttention(config, layer_type == SLIDING)
        self.mlp: AfmoeMLP | SwiGLU = (
            SwiGLU(config.hidden_size, config.intermediate_size)
            if layer < config.num_dense_layers
            else AfmoeMLP(config)
        )
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_mlp_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_mlp_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cache))
        return attended + self.post_mlp_layernorm(self.mlp(self.pre_mlp_layernorm(attended)))
