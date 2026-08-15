from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.gemma2.config import Gemma2Config
from mlx_omnia.engine.models.gemma2.layers.attention import Gemma2Attention

if TYPE_CHECKING:

    def _gelu(x: mx.array) -> mx.array: ...

else:
    _gelu = nn.gelu_approx


class Gemma2Block(nn.Module):
    def __init__(self, config: Gemma2Config, layer_type: str) -> None:
        super().__init__()
        self.self_attn = Gemma2Attention(config, layer_type)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, _gelu)
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cache))
        return attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )
