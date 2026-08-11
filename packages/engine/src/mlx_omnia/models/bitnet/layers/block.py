import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.models.bitnet.config import BitNetConfig
from mlx_omnia.models.bitnet.layers.attention import BitNetAttention
from mlx_omnia.models.bitnet.layers.mlp import BitNetMLP


class BitNetBlock(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class BitNetTrunk(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BitNetBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
