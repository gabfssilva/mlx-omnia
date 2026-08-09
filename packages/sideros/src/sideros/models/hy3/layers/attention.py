from sideros.core.attention import NormalizedFusedQKVAttention
from sideros.models.hy3.config import Hy3Config


class Hy3Attention(NormalizedFusedQKVAttention):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.theta,
            norm_eps=config.rms_norm_eps,
        )
