import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.cohere.config import CohereConfig
from mlx_omnia.engine.models.cohere.layers.attention import CohereAttention


class CohereBlock(nn.Module):
    def __init__(self, config: CohereConfig) -> None:
        super().__init__()
        self.self_attn = CohereAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias
        )

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        normed = self.input_layernorm(x)
        return x + self.self_attn(normed, cache) + self.mlp(normed)
