import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.models.apertus.config import ApertusConfig
from mlx_omnia.engine.models.apertus.layers.attention import ApertusAttention
from mlx_omnia.engine.models.apertus.layers.mlp import ApertusMLP


class ApertusBlock(nn.Module):
    def __init__(self, config: ApertusConfig) -> None:
        super().__init__()
        self.self_attn = ApertusAttention(config)
        self.mlp = ApertusMLP(config.hidden_size, config.intermediate_size)
        self.attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        attended = x + self.self_attn(self.attention_layernorm(x), cache)
        return attended + self.mlp(self.feedforward_layernorm(attended))
