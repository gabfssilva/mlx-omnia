import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.layers import split_qkv
from sideros.models.laguna.config import SLIDING, LagunaConfig, LagunaYaRNScaling


class LagunaAttention(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads_per_layer[layer_idx]
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding = config.layer_types[layer_idx] == SLIDING
        self.window = config.sliding_window if self.sliding else None
        rope = config.rope(layer_idx)
        self._rotary_dim = int(self.head_dim * rope.partial_rotary_factor)
        yarn = rope.yarn
        if yarn is not None:
            self._freqs, self._mscale = _yarn_freqs(
                self._rotary_dim, rope.rope_theta, yarn
            )
            mx.eval(self._freqs)
        else:
            self._freqs = None
            self._mscale = 1.0
            self._base = rope.rope_theta

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        queries = self._rope(self.q_norm(q), offset)
        rotated = self._rope(self.k_norm(k), offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, query_width)
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = (
            output.reshape(1, length, self.heads, self.head_dim) * gate[..., None]
        ).reshape(1, length, query_width)
        return self.o_proj(output)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        if self._freqs is not None:
            if self._mscale != 1.0:
                x = mx.concatenate(
                    [
                        x[..., : self._rotary_dim] * self._mscale,
                        x[..., self._rotary_dim :],
                    ],
                    axis=-1,
                )
            return mx.fast.rope(
                x,
                self._rotary_dim,
                traditional=False,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=self._freqs,
            )
        return mx.fast.rope(
            x, self._rotary_dim, traditional=False, base=self._base, scale=1.0, offset=offset
        )


def _yarn_freqs(
    rotary_dim: int, base: float, scaling: LagunaYaRNScaling
) -> tuple[mx.array, float]:
    """NTK-by-parts frequency table and the mscale that pre-scales q/k before the
    rotation. Same formula as the gpt_oss ``yarn_rope`` and mlx-vlm's ``YarnRoPE``."""
    factor = scaling.factor
    original = scaling.original_max_position_embeddings

    def correction(rotations: float) -> float:
        return (rotary_dim * math.log(original / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction(scaling.beta_fast)), 0)
    high = min(math.ceil(correction(scaling.beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001

    extra = base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
    inter = factor * extra
    ramp = mx.clip(
        (mx.arange(rotary_dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1
    )
    mask = 1.0 - ramp
    freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
    return freqs, scaling.attention_factor
