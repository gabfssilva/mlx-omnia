from typing import assert_never

import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import SwitchLinear
from sideros.models.deepseek_v4.config import LOCAL, OVERLAP, DeepseekV4Config
from sideros.models.deepseek_v4.layers.cache import DeepseekV4Cache
from sideros.models.deepseek_v4.layers.compressor import Compressor, Indexer
from sideros.models.deepseek_v4.layers.rope import rotary


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
        self, x: mx.array, mask: mx.array | str | None, cache: DeepseekV4Cache
    ) -> mx.array:
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
        out = self.wo_a(out.transpose(0, 1, 3, 2, 4).flatten(-2), mx.arange(self.o_groups)[None])
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
        if pool_mask is None:
            pool_mask = mx.ones((length, pooled.shape[2]), dtype=mx.bool_)
        dense = mx.concatenate([_dense(mask, length, keys.shape[2]), pool_mask], axis=-1)
        return mx.concatenate([keys, pooled], axis=2), dense


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
