import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.models.olmoe.config import OlmoEConfig
from mlx_omnia.engine.models.olmoe.layers.attention import OlmoEAttention
from mlx_omnia.engine.models.olmoe.layers.moe import OlmoEMLP


class OlmoEBlock(nn.Module):
    def __init__(self, config: OlmoEConfig) -> None:
        super().__init__()
        self.self_attn = OlmoEAttention(config)
        self.mlp = OlmoEMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))
