import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.models.gemma3n.config import Gemma3nTextConfig


class Laurel(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        self.linear_left = nn.Linear(config.hidden_size, config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(config.laurel_rank, config.hidden_size, bias=False)
        self.post_laurel_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.post_laurel_norm(self.linear_right(self.linear_left(x)))
