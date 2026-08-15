import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.models.gpt2.config import GPT2Config
from mlx_omnia.engine.models.gpt2.layers.attention import GPT2Attention
from mlx_omnia.engine.models.gpt2.layers.mlp import GPT2MLP


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def __call__(self, x: mx.array, cache: KVStore | None = None) -> mx.array:
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))
