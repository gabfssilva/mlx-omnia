import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.qwen3_next.config import Qwen3NextConfig
from mlx_omnia.engine.models.qwen3_next.layers.attention import Qwen3NextAttention
from mlx_omnia.engine.models.qwen3_next.layers.deltanet import Qwen3NextDeltaNet, Recurring
from mlx_omnia.engine.models.qwen3_next.layers.moe import Qwen3NextMoE

type Qwen3NextLayer = LayerCache | KVStore | Recurring
"""A layer's cache, alone or standing for one row each: attention reads it through
`core.attend`, and the DeltaNet through `window`/`state`."""


class Qwen3NextBlock(nn.Module):
    def __init__(self, config: Qwen3NextConfig, attends: bool, routes: bool) -> None:
        super().__init__()
        self.attends = attends
        if attends:
            self.self_attn = Qwen3NextAttention(config)
        else:
            self.linear_attn = Qwen3NextDeltaNet(config)
        self.mlp: Qwen3NextMoE | SwiGLU = (
            Qwen3NextMoE(config)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: Qwen3NextLayer) -> mx.array:
        normed = self.input_layernorm(x)
        # One mixer or the other; mlx.nn.Module's __getattr__ is untyped.
        if self.attends:
            attention = self.self_attn
            assert isinstance(attention, Qwen3NextAttention)
            assert isinstance(cache, LayerCache)
            mixed = x + attention(normed, cache)
        else:
            linear = self.linear_attn
            assert isinstance(linear, Qwen3NextDeltaNet) and isinstance(cache, Recurring)
            mixed = x + linear(normed, cache)
        return mixed + self.mlp(self.post_attention_layernorm(mixed))


class Qwen3NextTrunk(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3NextBlock(config, attends, routes)
            for attends, routes in zip(config.attends, config.routes, strict=True)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
