import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.layers import MultiLinear
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.engine.models.bailing_hybrid.layers.cache import BatchedLatentKVCache

type LatentStore = KVCache | BatchedLatentKVCache
"""The latent cache a layer reads: one sequence's, or one per row of a ragged batch."""


class BailingHybridLatentAttention(nn.Module):
    """MLA over a latent cache: absorbed on the single-token step, expanded on prefill."""

    def __init__(self, config: BailingHybridConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.rope_theta = config.rope_theta
        self.rope_traditional = config.rope_interleave
        self.scale = config.qk_head_dim**-0.5
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(config.qk_nope_head_dim, config.kv_lora_rank, self.heads)
        self.unembed_out = MultiLinear(config.kv_lora_rank, config.v_head_dim, self.heads)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False)
        self.dense = nn.Linear(self.heads * config.v_head_dim, hidden, bias=False)

    def rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return mx.fast.rope(
            x,
            self.qk_rope_head_dim,
            traditional=self.rope_traditional,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

    def __call__(self, x: mx.array, cache: LatentStore) -> mx.array:
        rows, length = x.shape[0], x.shape[1]
        offset = cache.offset
        q = (
            self.q_proj(x)
            .reshape(rows, length, self.heads, self.qk_head_dim)
            .transpose(0, 2, 1, 3)
        )
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        q_pe = self.rotate(q_pe, offset)

        projected = self.kv_a_proj_with_mqa(x)
        compressed, k_pe = mx.split(projected, [self.kv_lora_rank], axis=-1)
        k_pe = self.rotate(
            k_pe.reshape(rows, length, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3), offset
        )
        latent = mx.expand_dims(self.kv_a_layernorm(compressed), axis=1)

        if isinstance(cache, BatchedLatentKVCache):
            if length != 1:
                raise ValueError("a ragged batch decodes one token at a time")
            queries = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1)
            attended = mx.concatenate(
                [
                    self._absorbed(queries[index : index + 1], row_latent, row_pe)
                    for index, (row_latent, row_pe) in enumerate(
                        cache.update_and_fetch(latent, k_pe)
                    )
                ]
            )
            attended = self.unembed_out(attended)
        else:
            history, history_pe = cache.update_and_fetch(latent, k_pe)
            if length == 1:
                # Absorbed: the query moves into the latent's space, so nothing expands and
                # the whole prefix is read once, 576 columns wide.
                queries = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1)
                attended = self._absorbed(queries, history, history_pe)
                attended = self.unembed_out(attended)
            else:
                k_nope = self.embed_q(history, transpose=False)
                values = self.unembed_out(history)
                keys = mx.concatenate(
                    [k_nope, mx.repeat(history_pe, self.heads, axis=1)], axis=-1
                )
                attended = mx.fast.scaled_dot_product_attention(
                    mx.concatenate([q_nope, q_pe], axis=-1),
                    keys,
                    values,
                    scale=self.scale,
                    mask="causal",
                )

        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(attended.dtype)
        attended = attended * gate.transpose(0, 2, 1)[..., None]
        return self.dense(
            attended.transpose(0, 2, 1, 3).reshape(rows, length, self.heads * self.v_head_dim)
        )

    def _absorbed(self, queries: mx.array, latent: mx.array, latent_pe: mx.array) -> mx.array:
        """One absorbed step against one history: keys are `concat(latent, k_pe)`, values are
        the latent itself, so the output comes back in latent space."""
        return mx.fast.scaled_dot_product_attention(
            queries,
            mx.concatenate([latent, latent_pe], axis=-1),
            latent,
            scale=self.scale,
            mask=None,
        )
