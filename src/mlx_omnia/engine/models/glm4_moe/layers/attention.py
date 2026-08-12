from mlx_omnia.engine.core.attention import NormalizedFusedQKVAttention
from mlx_omnia.engine.models.glm4_moe.config import Glm4MoEConfig


class Glm4MoEAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: Glm4MoEConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_dims=config.rope_dims,
            norm_eps=config.rms_norm_eps,
            normalize=config.use_qk_norm,
            qkv_bias=config.attention_bias,
        )
