from sideros.core.attention import FusedQKVAttention
from sideros.models.ernie4_5_moe.config import Ernie45MoEConfig


class Ernie45MoEAttention(FusedQKVAttention):
    def __init__(self, config: Ernie45MoEConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_size,
            rope_theta=config.rope_theta,
            traditional=True,
        )
