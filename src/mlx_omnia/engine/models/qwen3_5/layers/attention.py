import math
from functools import cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_omnia.engine.core.attend import KVStore, attend
from mlx_omnia.engine.core.cache import FixedKVCache
from mlx_omnia.engine.models.qwen3_5.config import Qwen35TextConfig


@cache
def _mrope_sources(config: Qwen35TextConfig) -> mx.array:
    """Which of T/H/W each rotated dimension reads, as transformers' interleave lays it
    out: frequency `i` belongs to section `i % 3` until that section's quota runs out.
    Frequency `i` drives dims `i` and `i + rope_dims/2` (the half rotation); dims past
    `rope_dims` are not rotated at all, so their section never matters."""
    half = config.rope_dims // 2
    sections = np.zeros(half, dtype=np.int32)
    for section, count in enumerate(config.rope_parameters.mrope_section[1:], start=1):
        sections[section : 3 * count : 3] = section
    tail = np.zeros(config.head_dim - config.rope_dims, dtype=np.int32)
    return mx.array(np.concatenate([sections, sections, tail]))


class Qwen35Attention(nn.Module):
    """GQA whose q_proj emits `[query ‖ gate]` per head, with a 256-wide q/k-norm and
    a partial rope over the first quarter of the head."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.config = config
        queries = config.num_attention_heads * config.head_dim
        key_values = config.num_key_value_heads * config.head_dim
        self.fused_proj = nn.Linear(config.hidden_size, 2 * queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def rope(self, x: mx.array, offset: int | mx.array) -> mx.array:
        """Text-only MRoPE is a plain partial rope: the three sections read the same
        position, so the interleave rewrites each frequency with its own value."""
        config = self.config
        return mx.fast.rope(
            x, config.rope_dims, traditional=False, base=config.rope_parameters.rope_theta,
            scale=1.0, offset=offset,
        )

    def mrope(self, x: mx.array, positions: mx.array) -> mx.array:
        """Interleaved MRoPE over `[1, L, H, D]`, out in the same layout.

        Viewing `[1, L, H, D]` as `[L, H, 1, D]` is a free reshape (same element order)
        and turns each token into a batch row, which is what lets the rope kernel take
        one offset per token. The interleave then falls out of three calls to that same
        kernel and one select per dimension, without a single new floating-point
        operation — so the text path is identical to `rope` **by construction**, not by
        inspection. A hand-rolled cos/sin would not be: the kernel uses
        `metal::fast::cos` while mlx's elementwise `cos` uses `metal::precise::cos`.
        """
        batch, length, heads, dims = x.shape
        rows = x.reshape(batch * length, heads, 1, dims)
        config = self.config
        rotated = [
            mx.fast.rope(
                rows, config.rope_dims, traditional=False, base=config.rope_parameters.rope_theta,
                scale=1.0, offset=positions[section],
            )
            for section in range(3)
        ]
        source = _mrope_sources(self.config)
        picked = mx.where(source == 2, rotated[2], mx.where(source == 1, rotated[1], rotated[0]))
        return picked.reshape(batch, length, heads, dims)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        config = self.config
        rows, length = x.shape[0], x.shape[1]
        queries = config.num_attention_heads * config.head_dim
        key_values = config.num_key_value_heads * config.head_dim
        fused = self.fused_proj(x)
        q_gate, k, v = mx.split(fused, (2 * queries, 2 * queries + key_values), axis=-1)
        q_gate = q_gate.reshape(rows, length, config.num_attention_heads, 2 * config.head_dim)
        q, gate = mx.split(q_gate, (config.head_dim,), axis=-1)
        heads = (config.num_key_value_heads, config.head_dim)
        return (
            self.q_norm(q),
            self.k_norm(k.reshape(rows, length, *heads)),
            v.reshape(rows, length, *heads),
            gate.reshape(rows, length, queries),
        )

    def __call__(
        self,
        x: mx.array,
        cache: KVStore,
        positions: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        """`mask` is the fixed buffer's own fill, and only a compiled decode passes one: a
        growing cache holds exactly the rows written, so `None` attends all of them. A
        fixed buffer holds its whole capacity, and the columns past the position are
        zeros the softmax would otherwise weigh."""
        config = self.config
        length = x.shape[1]
        queries = config.num_attention_heads * config.head_dim
        # Read before `update_and_fetch` moves it: the rotation belongs to the row this
        # step is about to write. The fixed buffer answers with an array, which is what
        # keeps the offset an input of the trace instead of a constant baked at the first
        # token — `mx.fast.rope` takes either, and the two are bit-identical.
        offset = cache.position if isinstance(cache, FixedKVCache) else cache.offset
        q, k, v, gate = self.split_heads(x)
        if positions is None:
            q, k = self.rope(q.transpose(0, 2, 1, 3), offset), self.rope(
                k.transpose(0, 2, 1, 3), offset
            )
        else:
            q = self.mrope(q, positions).transpose(0, 2, 1, 3)
            k = self.mrope(k, positions).transpose(0, 2, 1, 3)
        attended = attend(
            cache,
            q,
            keys=k,
            values=v.transpose(0, 2, 1, 3),
            scale=1 / math.sqrt(config.head_dim),
            # An explicit mask wins at any length: the compiled verify feeds `rows` rows
            # over a fixed buffer, where the fill mask is also the causal one. Without
            # one the growing paths keep their answers — all rows at T=1, causal past it.
            mask=mask if mask is not None else (None if length == 1 else "causal"),
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(x.shape[0], length, queries)
        return self.o_proj(attended * mx.sigmoid(gate))
