from sideros.core.attention import FusedQKVAttention
from sideros.models.seed_oss.config import SeedOssConfig


class SeedOssAttention(FusedQKVAttention):
    def __init__(self, config: SeedOssConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=config.attention_bias,
            output_bias=config.attention_out_bias,
        )
