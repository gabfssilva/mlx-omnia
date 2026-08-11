from sideros.core.attention import FusedQKVAttention
from sideros.models.qwen2.config import Qwen2Config


class Qwen2Attention(FusedQKVAttention):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=True,
        )
