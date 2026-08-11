from mlx_omnia.core.attention import FusedQKVAttention
from mlx_omnia.models.falcon_h1.config import FalconH1Config


class FalconH1Attention(FusedQKVAttention):
    def __init__(self, config: FalconH1Config) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=config.attention_bias,
            output_bias=config.attention_bias,
        )
        self.config = config
