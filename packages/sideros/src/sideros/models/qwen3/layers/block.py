import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.add_rms_norm import add_rms_norm, add_rms_norm_applies
from sideros.core.layers import SwiGLU
from sideros.models.qwen3.config import Qwen3Config, Qwen3MoEConfig
from sideros.models.qwen3.layers import flags
from sideros.models.qwen3.layers.attention import Qwen3Attention, Qwen3MoEAttention
from sideros.models.qwen3.layers.moe import Qwen3MoEMLP


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen3MoEBlock(nn.Module):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3MoEAttention(config)
        self.mlp = Qwen3MoEMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden_size

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended, h = self._join(x, cache)
        if x.shape[1] == 1 and self.mlp.step_applies():
            return self.mlp.step(h, attended)
        return attended + self.mlp(h)

    def _join(self, x: mx.array, cache: KVCache) -> tuple[mx.array, mx.array]:
        """(x + attention, its post-norm). At T=1 one kernel does the pair."""
        if flags.ADD_RMS_NORM_KERNEL and x.shape[1] == 1 and add_rms_norm_applies(self.hidden):
            normed = mx.fast.rms_norm(x, self.input_layernorm.weight, self.eps)
            return add_rms_norm(
                x,
                self.self_attn(normed, cache),
                self.post_attention_layernorm.weight,
                self.eps,
            )
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended, self.post_attention_layernorm(attended)
