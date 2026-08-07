import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.models.gpt2.config import GPT2Config


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = (
            part.reshape(batch, length, self.n_head, -1).transpose(0, 2, 1, 3)
            for part in mx.split(qkv, 3, axis=-1)
        )
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        # A lone query attends to everything: no mask on the T=1 step.
        mask = "causal" if length > 1 else None
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask=mask
        )
        return self.c_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))
