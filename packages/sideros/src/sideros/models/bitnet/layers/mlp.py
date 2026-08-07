import mlx.core as mx
import mlx.nn as nn

from sideros.models.bitnet.config import BitNetConfig
from sideros.models.bitnet.layers.bitlinear import BitLinear


class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.intermediate_size
        self.gate_proj = BitLinear(hidden, inner)
        self.up_proj = BitLinear(hidden, inner)
        self.down_proj = BitLinear(inner, hidden)
        self.ffn_sub_norm = nn.RMSNorm(inner, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.ffn_sub_norm(_relu2(self.gate_proj(x)) * self.up_proj(x)))


def _relu2(x: mx.array) -> mx.array:
    gated = mx.maximum(x, mx.array(0.0, dtype=x.dtype))
    return gated * gated
