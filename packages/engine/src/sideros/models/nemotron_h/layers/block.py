from typing import assert_never

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache, KVCache, LayerCache
from sideros.models.nemotron_h.config import BlockKind, NemotronHConfig
from sideros.models.nemotron_h.layers.attention import NemotronHAttention
from sideros.models.nemotron_h.layers.mamba import NemotronHMamba
from sideros.models.nemotron_h.layers.mlp import NemotronHMLP
from sideros.models.nemotron_h.layers.moe import NemotronHMoE


class NemotronHBlock(nn.Module):
    def __init__(self, config: NemotronHConfig, block_type: BlockKind) -> None:
        super().__init__()
        self.block_type = block_type
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        match block_type:
            case "M":
                self.mixer = NemotronHMamba(config)
            case "*":
                self.mixer = NemotronHAttention(config)
            case "E":
                self.mixer = NemotronHMoE(config)
            case "-":
                self.mixer = NemotronHMLP(
                    config.hidden_size, config.intermediate_size, config.mlp_bias
                )
            case _:
                assert_never(block_type)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        normed = self.norm(x)
        kind: BlockKind = self.block_type
        # One mixer kind per layer; mlx.nn.Module's __getattr__ is untyped, so each
        # branch narrows what it holds.
        match kind:
            case "M":
                mixer = self.mixer
                assert isinstance(mixer, NemotronHMamba) and isinstance(cache, DeltaCache)
                return x + mixer(normed, cache)
            case "*":
                attention = self.mixer
                assert isinstance(attention, NemotronHAttention) and isinstance(cache, KVCache)
                return x + attention(normed, cache)
            case "E" | "-":
                stateless = self.mixer
                assert isinstance(stateless, NemotronHMoE | NemotronHMLP)
                cache.offset += x.shape[1]
                return x + stateless(normed)
            case _:
                assert_never(kind)
