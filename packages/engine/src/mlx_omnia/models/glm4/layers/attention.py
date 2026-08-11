from mlx_omnia.core.attention import FusedQKVAttention
from mlx_omnia.models.glm4.config import Glm4Config


class Glm4Attention(FusedQKVAttention):
    def __init__(self, config: Glm4Config) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_dims=config.rope_dims,
            traditional=True,
            qkv_bias=config.attention_bias,
        )
