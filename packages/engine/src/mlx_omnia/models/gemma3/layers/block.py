from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.core.layers import SwiGLU
from mlx_omnia.models.gemma3.config import Gemma3TextConfig
from mlx_omnia.models.gemma3.layers.attention import Gemma3Attention

# `hidden_activation` is gelu_pytorch_tanh: the tanh approximation, which is what
# mlx.nn.gelu_approx computes — as a single compiled kernel, hence the reuse instead of
# the expression inline. `mx.compile` erases the signature from the stubs, so it is
# restated here the way `core/mxcompat.py` restates mlx's own stale ones.
if TYPE_CHECKING:

    def _gelu(x: mx.array) -> mx.array: ...

else:
    _gelu = nn.gelu_approx


class Gemma3Block(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_type)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, _gelu)
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cache))
        return attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )


class Gemma3Trunk(nn.Module):
    def __init__(self, config: Gemma3TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3Block(config, layer_type) for layer_type in config.layer_types]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
