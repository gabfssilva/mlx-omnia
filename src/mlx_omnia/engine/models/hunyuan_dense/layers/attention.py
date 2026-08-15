from mlx_omnia.engine.core.attention import NormalizedFusedQKVAttention
from mlx_omnia.engine.models.hunyuan_dense.config import HunyuanDenseConfig


class HunyuanDenseAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: HunyuanDenseConfig) -> None:
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.kv_heads,
            head_dim=config.head_size,
            rope_theta=config.rope_base,
            norm_eps=config.rms_norm_eps,
            norm_names="query_layernorm",
            qkv_bias=config.attention_bias,
            output_bias=config.attention_bias,
        )
