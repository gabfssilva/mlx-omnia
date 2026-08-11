import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import SwiGLU
from sideros.models.hunyuan_dense.config import HunyuanDenseConfig
from sideros.models.hunyuan_dense.layers.attention import HunyuanDenseAttention


class HunyuanDenseBlock(nn.Module):
    def __init__(self, config: HunyuanDenseConfig) -> None:
        super().__init__()
        self.self_attn = HunyuanDenseAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class HunyuanDenseTrunk(nn.Module):
    def __init__(self, config: HunyuanDenseConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [HunyuanDenseBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
