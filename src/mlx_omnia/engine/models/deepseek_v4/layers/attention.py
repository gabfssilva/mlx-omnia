from typing import assert_never

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.layers import SwitchLinear
from mlx_omnia.engine.models.deepseek_v4.config import LOCAL, OVERLAP, DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.cache import (
    DeepseekV4Cache,
    DeepseekV4Solo,
    FixedDeepseekV4Cache,
)
from mlx_omnia.engine.models.deepseek_v4.layers.compressor import Compressor, Indexer
from mlx_omnia.engine.models.deepseek_v4.layers.rope import rotary


class DeepseekV4Attention(nn.Module):
    """MQA with `head_dim` 512 and K == V, over the last 128 tokens plus — on the
    compressed layers — the whole pooled history."""

    def __init__(self, config: DeepseekV4Config, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.eps = config.rms_norm_eps
        self.window = config.sliding_window
        self.scale = self.head_dim**-0.5
        hidden = config.hidden_size
        self.wq_a = nn.Linear(hidden, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(hidden, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = SwitchLinear(
            config.o_groups,
            self.heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
        )
        # The footprint walk asks the module above a stack how many slots a token reaches;
        # block-diagonal means all of them, on every token, and no routing decides.
        self.k = config.o_groups
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank, hidden, bias=config.attention_bias
        )
        self.attn_sink = mx.zeros((self.heads,), dtype=mx.float32)
        self._groups = mx.arange(config.o_groups)[None]
        self.rope = rotary(
            config.qk_rope_head_dim,
            self.head_dim,
            config.rope_theta if ratio == LOCAL else config.compress_rope_theta,
            None if ratio == LOCAL else config.rope_scaling,
        )
        # Declared as absent and only then built: a leaf the layer does not have is a leaf
        # the checkpoint does not carry either, and `load_weights(strict=True)` says so.
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if ratio != LOCAL:
            self.compressor = Compressor(config, ratio, self.head_dim)
        if ratio == OVERLAP:
            self.indexer = Indexer(config, ratio)

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: DeepseekV4Solo
    ) -> mx.array:
        if isinstance(cache, FixedDeepseekV4Cache):
            return self._step(x, cache)
        length = x.shape[1]
        offset = cache.attention.offset
        residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(residual).reshape(1, length, self.heads, self.head_dim)
        q = self.rope(mx.fast.rms_norm(q, weight=None, eps=self.eps).transpose(0, 2, 1, 3), offset)

        kv = self.kv_norm(self.wkv(x)).reshape(1, 1, length, self.head_dim)
        keys, _ = cache.attention.update_and_fetch(
            self.rope(kv, offset), mx.zeros((1, 1, length, 0), dtype=kv.dtype)
        )
        if self.ratio != LOCAL:
            keys, mask = self._columns(x, residual, cache, offset, keys, mask)

        attended = mx.fast.scaled_dot_product_attention(
            q, keys, keys, scale=self.scale, mask=mask, sinks=self.attn_sink.astype(q.dtype)
        )
        out = self.rope(attended, offset, inverse=True)
        out = out.reshape(1, self.o_groups, -1, length, self.head_dim)
        out = self.wo_a(out.transpose(0, 1, 3, 2, 4).flatten(-2), self._groups)
        return self.wo_b(out.transpose(0, 2, 1, 3).flatten(-2))

    def _columns(
        self,
        x: mx.array,
        residual: mx.array,
        cache: DeepseekV4Cache,
        offset: int,
        keys: mx.array,
        mask: mx.array | str | None,
    ) -> tuple[mx.array, mx.array | str | None]:
        """The pooled rows appended to the local keys, and the mask that goes with them."""
        length = x.shape[1]
        assert cache.compressor is not None and self.compressor is not None
        pooled = self.compressor(x, cache.compressor, offset)
        # Unconditionally, and before the early return: the indexer keeps a second pool over
        # the same windows, and a token skipped there shifts every window it holds against
        # the ones it is supposed to be selecting from.
        selected = (
            self.indexer(x, residual, cache.indexer, offset)
            if self.indexer is not None and cache.indexer is not None
            else None
        )
        if not pooled.shape[2]:
            return keys, mask
        pool_mask = cache.compressor.mask(length, offset)
        if selected is not None:
            sparse = mx.put_along_axis(
                mx.zeros((length, pooled.shape[2]), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
            pool_mask = sparse if pool_mask is None else sparse & pool_mask
        if pool_mask is None and mask is None:
            # Every pooled row and every local key is behind the single query: an
            # all-true mask, which the no-mask sdpa path handles without building it.
            return mx.concatenate([keys, pooled], axis=2), None
        if pool_mask is None:
            pool_mask = mx.ones((length, pooled.shape[2]), dtype=mx.bool_)
        dense = mx.concatenate([_dense(mask, length, keys.shape[2]), pool_mask], axis=-1)
        return mx.concatenate([keys, pooled], axis=2), dense


    def _step(self, x: mx.array, cache: FixedDeepseekV4Cache) -> mx.array:
        """One token over a fixed cache, with every position read off the graph.

        The body is `__call__`'s with `offset` an `mx.array` instead of a host int, which is
        the whole of what the trace demands of the arithmetic. What it costs elsewhere is the
        mask: the trunk cannot build the sliding band from a stale host offset, so it is
        built here off the same tensor, and the pooled columns' visibility comes from the
        pool's own rule rather than from how many rows it happens to hold.
        """
        position = cache.position
        residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(residual).reshape(1, 1, self.heads, self.head_dim)
        q = self.rope(
            mx.fast.rms_norm(q, weight=None, eps=self.eps).transpose(0, 2, 1, 3), position
        )

        kv = self.kv_norm(self.wkv(x)).reshape(1, 1, 1, self.head_dim)
        keys, _ = cache.attention.update_and_fetch(
            self.rope(kv, position), mx.zeros((1, 1, 1, 0), dtype=kv.dtype)
        )
        mask = self._band(position, keys.shape[2])
        if self.ratio != LOCAL:
            keys, mask = self._fixed_columns(x, residual, cache, position, keys, mask)

        attended = mx.fast.scaled_dot_product_attention(
            q, keys, keys, scale=self.scale, mask=mask, sinks=self.attn_sink.astype(q.dtype)
        )
        out = self.rope(attended, position, inverse=True)
        out = out.reshape(1, self.o_groups, -1, 1, self.head_dim)
        out = self.wo_a(out.transpose(0, 1, 3, 2, 4).flatten(-2), self._groups)
        return self.wo_b(out.transpose(0, 2, 1, 3).flatten(-2))

    def _band(self, position: mx.array, span: int) -> mx.array:
        """The trunk's window rule over a fixed buffer: `column <= position` cuts the columns
        the buffer has not written, `column > position - window` is the slide. One expression
        for both regimes, which is what the trunk's host-side `_window` cannot be here."""
        columns = mx.arange(span).reshape(1, -1)
        return (columns <= position) & (columns > position - self.window)

    def _fixed_columns(
        self,
        x: mx.array,
        residual: mx.array,
        cache: FixedDeepseekV4Cache,
        position: mx.array,
        keys: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """`_columns` over the fixed pools: the buffer is capacity-sized and always
        non-empty, so what was a host branch on how many rows had been pooled is the mask."""
        assert cache.compressor is not None and self.compressor is not None
        pooled = self.compressor.step(x, cache.compressor, position)
        selected = (
            self.indexer.step(x, residual, cache.indexer, position)
            if self.indexer is not None and cache.indexer is not None
            else None
        )
        pool_mask = cache.compressor.mask(1, position)
        if selected is not None:
            sparse = mx.put_along_axis(
                mx.zeros((1, pooled.shape[2]), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
            pool_mask = sparse & pool_mask
        return mx.concatenate([keys, pooled], axis=2), mx.concatenate(
            [mask, pool_mask], axis=-1
        )


def _dense(mask: mx.array | str | None, length: int, total: int) -> mx.array:
    """The local mask as an array, for the layers that concatenate pooled columns to it."""
    match mask:
        case mx.array():
            return mask
        case None:
            return mx.ones((length, total), dtype=mx.bool_)
        case str():
            rows = mx.arange(total - length, total).reshape(-1, 1)
            return rows >= mx.arange(total).reshape(1, -1)
        case _:
            assert_never(mask)
