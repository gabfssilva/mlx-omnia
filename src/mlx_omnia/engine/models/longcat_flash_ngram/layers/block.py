from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.longcat_flash_ngram.config import LongcatFlashNgramConfig
from mlx_omnia.engine.models.longcat_flash_ngram.layers.attention import LongcatFlashMLA
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import LatentStore
from mlx_omnia.engine.models.longcat_flash_ngram.layers.moe import LongcatFlashMoE


class LongcatFlashDecoderLayer(nn.Module):
    def __init__(
        self, config: LongcatFlashNgramConfig, freqs: mx.array, mscale: float
    ) -> None:
        super().__init__()
        self.mlp = LongcatFlashMoE(config)
        self.self_attn = [
            LongcatFlashMLA(config, freqs, mscale) for _ in range(2)
        ]
        self.mlps = [
            SwiGLU(config.hidden_size, config.ffn_hidden_size)
            for _ in range(2)
        ]
        self.input_layernorm = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self, x: mx.array, mask: mx.array | None, cache: Sequence[LatentStore]
    ) -> mx.array:
        shortcut = None
        for i in range(2):
            residual = x
            x = self.input_layernorm[i](x)
            x = self.self_attn[i](x, mask, cache[i])
            x = residual + x

            residual = x
            x = self.post_attention_layernorm[i](x)
            if i == 0:
                shortcut = self.mlp(x)
            x = self.mlps[i](x)
            x = residual + x
            if i == 1:
                assert shortcut is not None
                x = x + shortcut
        return x
