import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache, KVCache, LayerCache
from sideros.core.layers import SwiGLU
from sideros.models.jamba.config import JambaConfig
from sideros.models.jamba.layers.attention import JambaAttention
from sideros.models.jamba.layers.mamba import JambaMamba
from sideros.models.jamba.layers.moe import JambaMoE


class JambaBlock(nn.Module):
    def __init__(self, config: JambaConfig, attends: bool, routes: bool) -> None:
        super().__init__()
        self.attends = attends
        if attends:
            self.self_attn = JambaAttention(config)
        else:
            self.mamba = JambaMamba(config)
        self.feed_forward: JambaMoE | SwiGLU = (
            JambaMoE(config)
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        normed = self.input_layernorm(x)
        # One mixer or the other; mlx.nn.Module's __getattr__ is untyped, so the branch
        # is narrowed here.
        if self.attends:
            attention = self.self_attn
            assert isinstance(attention, JambaAttention) and isinstance(cache, KVCache)
            mixed = x + attention(normed, cache)
        else:
            mamba = self.mamba
            assert isinstance(mamba, JambaMamba) and isinstance(cache, DeltaCache)
            mixed = x + mamba(normed, cache)
        return mixed + self.feed_forward(self.pre_ff_layernorm(mixed))
