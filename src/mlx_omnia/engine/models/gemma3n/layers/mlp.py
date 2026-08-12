import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.models.gemma3n.config import Gemma3nTextConfig

if TYPE_CHECKING:

    def gelu(x: mx.array) -> mx.array: ...

else:
    gelu = nn.gelu_approx


class Gemma3nMLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer: int) -> None:
        super().__init__()
        inner = config.mlp_widths[layer]
        self.gate_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inner, bias=False)
        self.down_proj = nn.Linear(inner, config.hidden_size, bias=False)
        sparsity = config.activation_sparsity[layer]
        self.cutoff = (
            math.sqrt(2.0) * float(mx.erfinv(mx.array(2 * sparsity - 1)).item())
            if sparsity > 0
            else None
        )

    def activate(self, gate: mx.array) -> mx.array:
        if self.cutoff is None:
            return gelu(gate)
        mean = mx.mean(gate, axis=-1, keepdims=True)
        std = mx.std(gate, axis=-1, keepdims=True)
        return gelu(mx.maximum(0, gate - (mean + std * self.cutoff)))

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.activate(self.gate_proj(x)) * self.up_proj(x))
