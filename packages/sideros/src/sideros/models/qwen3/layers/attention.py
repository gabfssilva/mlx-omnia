import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.rope_epilogue import rope_epilogue, rope_epilogue_applies
from sideros.core.layers import split_qkv
from sideros.models.qwen3.config import Qwen3Config, Qwen3MoEConfig
from sideros.models.qwen3.layers import flags


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config | Qwen3MoEConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.eps = config.rms_norm_eps
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """qkv split back into per-head [1, heads, length, head_dim], normed but
        unrotated — the boundary transformers' q_norm/k_norm hooks expose."""
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        return self.q_norm(q), self.k_norm(k), v

    def rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def step_heads(self, x: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        """T=1: q/k norm + rotation in one dispatch, v a free slice of the projection."""
        fused = self.qkv_proj(x)
        queries, keys = rope_epilogue(
            fused,
            query_heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm.weight,
            k_norm=self.k_norm.weight,
            offset=offset,
            base=self.rope_theta,
            eps=self.eps,
        )
        values = fused[..., (self.heads + self.kv_heads) * self.head_dim :]
        return (
            queries.reshape(1, self.heads, 1, self.head_dim),
            keys.reshape(1, self.kv_heads, 1, self.head_dim),
            values.reshape(1, self.kv_heads, 1, self.head_dim),
        )

    def step_applies(self, length: int) -> bool:
        return length == 1 and rope_epilogue_applies(self.head_dim)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        if self.step_applies(length):
            queries, rotated, v = self.step_heads(x, offset)
        else:
            q, k, v = self.split_heads(x)
            queries, rotated = self.rope(q, offset), self.rope(k, offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=1 / math.sqrt(self.head_dim),
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        )


class Qwen3MoEAttention(Qwen3Attention):
    def step_applies(self, length: int) -> bool:
        return flags.ROPE_EPILOGUE_KERNEL and super().step_applies(length)
