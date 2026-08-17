import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.masks import causal_mask
from mlx_omnia.engine.core.rope import Yarn
from mlx_omnia.engine.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from mlx_omnia.engine.models.glm4_moe.dsa.layers.cache import (
    BatchedDSACache,
    DSASolo,
    DSAStore,
    FixedDSACache,
)
from mlx_omnia.engine.models.glm4_moe.dsa.layers.indexer import Indexer


class GlmMoEDSAAttention(nn.Module):
    def __init__(self, config: GlmMoEDSAConfig, rope: Yarn) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.q_head_dim
        self.rope_theta = config.theta
        self.rope = rope
        self.scale = self.q_head_dim**-0.5 * rope.scale_correction
        hidden = config.hidden_size
        self.q_a_proj = nn.Linear(hidden, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.heads * config.v_head_dim, hidden, bias=config.attention_bias
        )
        self.indexer = Indexer(config, rope)

    def rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        if self.rope.freqs is None:
            return mx.fast.rope(
                x, self.qk_rope_head_dim, traditional=True, base=self.rope_theta,
                scale=1.0, offset=offset,
            )
        scaled = x * self.rope.mscale if self.rope.mscale != 1.0 else x
        return mx.fast.rope(
            scaled, self.qk_rope_head_dim, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=self.rope.freqs,
        )

    def __call__(self, x: mx.array, cache: DSAStore) -> mx.array:
        """A ragged batch runs the body once per row: the indexer's selection is a function
        of that row's history length — including whether there is anything to select at all —
        so there is no shape the rows share. Row by row it is the solo path by construction."""
        if isinstance(cache, BatchedDSACache):
            return mx.concatenate(
                [
                    self.forward(x[index : index + 1], row)
                    for index, row in enumerate(cache.sequences)
                ]
            )
        return self.forward(x, cache)

    def forward(self, x: mx.array, cache: DSASolo) -> mx.array:
        rows = x.shape[0]
        length = x.shape[1]
        # Read before either half is updated: a promoted cache answers with a graph tensor,
        # and both the attention's rotation and the indexer's take it.
        offset = cache.position if isinstance(cache, FixedDSACache) else cache.offset
        qr = self.q_a_layernorm(self.q_a_proj(x))
        lifted = self.q_b_proj(qr).reshape(rows, length, self.heads, self.q_head_dim)
        q = lifted.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        latent = self.kv_a_proj_with_mqa(x)
        compressed, k_pe = mx.split(latent, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(rows, length, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed))
            .reshape(rows, length, self.heads, -1)
            .transpose(0, 2, 1, 3)
        )
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        rotated = mx.repeat(self.rotate(k_pe, offset), self.heads, axis=1)
        keys, values = cache.attention.update_and_fetch(
            mx.concatenate([k_nope, rotated], axis=-1), v
        )

        total = keys.shape[2]
        # `total` is the rows written for a growing cache and the whole capacity for a
        # promoted one, so the band has to come from the cache rather than from the shape.
        # `readable` hands a growing cache's mask back untouched — `None` at T=1, the causal
        # band over a prefill — and cuts a fixed buffer's to the columns it has written.
        band = None if length == 1 else causal_mask(length, total, None)
        dense = cache.attention.readable(band, length)
        assert not isinstance(dense, str)
        # The indexer selects over the same columns, out of a buffer that advanced with
        # this one, so the band that says which of them exist is the same band.
        selected = self.indexer(x, qr, dense, cache.index)
        mask: mx.array | str | None
        if selected is None:
            mask = dense
        else:
            sparse = mx.put_along_axis(
                mx.zeros((rows, 1, length, total), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
            # The AND is what closes the regime where fewer columns are written than
            # `topk`: the selection then fills its remaining slots with columns the mask
            # scored at `-inf`, and those are exactly the ones `dense` drops. A fixed
            # buffer is in that regime for the first `topk` rows of every conversation.
            mask = sparse if dense is None else sparse & dense

        attended = mx.fast.scaled_dot_product_attention(
            mx.concatenate([q_nope, self.rotate(q_pe, offset)], axis=-1),
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(rows, length, self.heads * self.v_head_dim)
        )
