import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.masks import causal_mask
from mlx_omnia.core.rope import Yarn
from mlx_omnia.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from mlx_omnia.models.glm4_moe.dsa.layers.cache import DSACache
from mlx_omnia.models.glm4_moe.dsa.layers.indexer import Indexer


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

    def rotate(self, x: mx.array, offset: int) -> mx.array:
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

    def __call__(self, x: mx.array, cache: DSACache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(1, length, self.heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        latent = self.kv_a_proj_with_mqa(x)
        compressed, k_pe = mx.split(latent, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(1, length, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed))
            .reshape(1, length, self.heads, -1)
            .transpose(0, 2, 1, 3)
        )
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)
        rotated = mx.repeat(self.rotate(k_pe, offset), self.heads, axis=1)
        keys, values = cache.attention.update_and_fetch(
            mx.concatenate([k_nope, rotated], axis=-1), v
        )

        total = keys.shape[2]
        dense = None if length == 1 else causal_mask(length, total, None)
        selected = self.indexer(x, qr, dense, cache.index)
        mask: mx.array | str | None
        if selected is None:
            mask = dense
        else:
            sparse = mx.put_along_axis(
                mx.zeros((1, 1, length, total), dtype=mx.bool_),
                selected,
                mx.array(True),
                axis=-1,
            )
            mask = sparse if dense is None else sparse & dense

        attended = mx.fast.scaled_dot_product_attention(
            mx.concatenate([q_nope, self.rotate(q_pe, offset)], axis=-1),
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.v_head_dim)
        )
