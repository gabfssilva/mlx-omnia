import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore, attend
from mlx_omnia.engine.core.rope import Yarn
from mlx_omnia.engine.models.deepseek_v2.config import DeepseekV2Config


class DeepseekV2Attention(nn.Module):
    def __init__(self, config: DeepseekV2Config, rope: Yarn) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.q_head_dim
        self.rope_theta = config.rope_theta
        self.rope = rope
        self.scale = self.q_head_dim**-0.5 * rope.scale_correction
        hidden = config.hidden_size
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(hidden, self.heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.heads * self.q_head_dim, bias=False
            )
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

    def rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        if self.rope.freqs is None:
            return mx.fast.rope(
                x,
                self.qk_rope_head_dim,
                traditional=True,
                base=self.rope_theta,
                scale=1.0,
                offset=offset,
            )
        scaled = x * self.rope.mscale if self.rope.mscale != 1.0 else x
        return mx.fast.rope(
            scaled,
            self.qk_rope_head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self.rope.freqs,
        )

    def queries(self, x: mx.array) -> mx.array:
        if self.q_lora_rank is None:
            return self.q_proj(x)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        rows, length = x.shape[0], x.shape[1]
        offset = cache.offset
        q = self.queries(x).reshape(rows, length, self.heads, self.q_head_dim).transpose(0, 2, 1, 3)
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
        attended = attend(
            cache,
            mx.concatenate([q_nope, self.rotate(q_pe, offset)], axis=-1),
            keys=mx.concatenate([k_nope, rotated], axis=-1),
            values=v,
            scale=self.scale,
            mask=None if length == 1 else "causal",
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(rows, length, self.heads * self.v_head_dim)
        )
