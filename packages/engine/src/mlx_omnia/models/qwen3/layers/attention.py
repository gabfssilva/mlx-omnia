from mlx_omnia.core.attention import NormalizedFusedQKVAttention
from mlx_omnia.models.qwen3.config import Qwen3Config, Qwen3MoEConfig
from mlx_omnia.models.qwen3.layers import flags


class Qwen3Attention(NormalizedFusedQKVAttention):
    def __init__(self, config: Qwen3Config | Qwen3MoEConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            norm_eps=config.rms_norm_eps,
            fused_decode=True,
        )


class Qwen3MoEAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: Qwen3MoEConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            norm_eps=config.rms_norm_eps,
            fused_decode=flags.ROPE_EPILOGUE_KERNEL,
        )
