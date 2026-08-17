from typing import assert_never

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.mimo_v2.config import LayerType, MimoV2Config, MlpType
from mlx_omnia.engine.models.mimo_v2.layers.attention import MimoV2Attention
from mlx_omnia.engine.models.mimo_v2.layers.moe import MimoV2MoE


class MimoV2Block(nn.Module):
    def __init__(self, config: MimoV2Config, layer_type: LayerType, mlp_type: MlpType) -> None:
        super().__init__()
        self.self_attn = MimoV2Attention(config, layer_type)
        self.mlp: MimoV2MoE | SwiGLU = _mlp(config, mlp_type)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


def _mlp(config: MimoV2Config, mlp_type: MlpType) -> MimoV2MoE | SwiGLU:
    match mlp_type:
        case "sparse":
            return MimoV2MoE(config)
        case "dense":
            return SwiGLU(config.hidden_size, config.intermediate_size)
        case _:
            assert_never(mlp_type)
