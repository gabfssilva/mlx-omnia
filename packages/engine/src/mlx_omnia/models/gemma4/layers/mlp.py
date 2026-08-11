import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.models.gemma4.config import Gemma4TextConfig
from mlx_omnia.models.gemma4.layers.activation import gelu


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        inner = config.intermediate_for_layer(layer_idx)
        self.inner = inner
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(gelu(self.gate_proj(x)) * self.up_proj(x))
