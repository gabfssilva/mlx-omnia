from sideros.core.attention import FusedQKVAttention
from sideros.models.gemma.config import GemmaConfig


class GemmaAttention(FusedQKVAttention):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
        )
