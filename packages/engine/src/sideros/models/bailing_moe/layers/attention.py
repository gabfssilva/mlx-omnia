from sideros.core.attention import NormalizedFusedQKVAttention
from sideros.models.bailing_moe.config import BailingMoEConfig


class BailingMoEAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: BailingMoEConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_dims=config.rope_dims,
            norm_eps=config.rms_norm_eps,
            norm_names="query_layernorm",
            normalize=config.use_qk_norm,
            qkv_bias=config.use_qkv_bias,
            output_bias=config.use_bias,
            projection_name="query_key_value",
            output_name="dense",
        )
