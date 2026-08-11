import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import SwiGLU
from mlx_omnia.models.falcon_h1.config import FalconH1Config
from mlx_omnia.models.falcon_h1.layers.attention import FalconH1Attention
from mlx_omnia.models.falcon_h1.layers.cache import FalconH1LayerCache
from mlx_omnia.models.falcon_h1.layers.mamba import FalconH1Mixer


class FalconH1DecoderLayer(nn.Module):
    """Parallel mamba + attention on the same normalized input, summed; then
    the SwiGLU MLP."""

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.mamba = FalconH1Mixer(config)
        self.self_attn = FalconH1Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: FalconH1LayerCache) -> mx.array:
        h = self.input_layernorm(x)
        mamba_h = self.mamba(h, cache.mamba)
        attn_h = self.self_attn(h, cache.kv)
        h = x + mamba_h + attn_h
        return h + self.mlp(self.pre_ff_layernorm(h))


class FalconH1Trunk(nn.Module):
    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [FalconH1DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
