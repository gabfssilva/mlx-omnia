from mlx_omnia.core.attention import FusedQKVAttention
from mlx_omnia.models.qwen2_moe.config import Qwen2MoEConfig


class Qwen2MoEAttention(FusedQKVAttention):
    def __init__(self, config: Qwen2MoEConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=True,
        )
