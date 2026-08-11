import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.models.gpt_oss.config import GPTOSSConfig
from mlx_omnia.models.gpt_oss.layers.attention import GPTOSSAttention
from mlx_omnia.models.gpt_oss.layers.moe import GPTOSSMLP


class GPTOSSBlock(nn.Module):
    def __init__(self, config: GPTOSSConfig, freqs: mx.array, mscale: float) -> None:
        super().__init__()
        self.self_attn = GPTOSSAttention(config, freqs, mscale)
        self.mlp = GPTOSSMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        normed = self.post_attention_layernorm(attended)
        fused = self.mlp.fused_step(normed, attended)
        return fused if fused is not None else attended + self.mlp(normed)
