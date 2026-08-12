import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.seed_oss.config import SeedOssConfig
from mlx_omnia.engine.models.seed_oss.layers.attention import SeedOssAttention


class SeedOssBlock(nn.Module):
    def __init__(self, config: SeedOssConfig) -> None:
        super().__init__()
        self.self_attn = SeedOssAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class SeedOssTrunk(nn.Module):
    def __init__(self, config: SeedOssConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [SeedOssBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
