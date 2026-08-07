import mlx.core as mx
import mlx.nn as nn

from sideros.models.longcat_flash_ngram.config import LongcatFlashNgramConfig
from sideros.models.longcat_flash_ngram.layers.attention import LongcatFlashMLA
from sideros.models.longcat_flash_ngram.layers.cache import MLACache
from sideros.models.longcat_flash_ngram.layers.moe import LongcatFlashMoE


class LongcatFlashMLP(nn.Module):
    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self.inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.gate_up_proj(x)
        gate, up = mx.split(fused, [self.inner], axis=-1)
        return self.down_proj(mx.sigmoid(gate) * gate * up)


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
            LongcatFlashMLP(config.hidden_size, config.ffn_hidden_size)
            for _ in range(2)
        ]
        self.input_layernorm = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self, x: mx.array, mask: mx.array | None, cache: list[MLACache]
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
