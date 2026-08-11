import math

import mlx.core as mx
import mlx.nn as nn

from sideros.models.gpt2.config import GPT2Config


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(_gelu_new(self.c_fc(x)))


def _gelu_new(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
