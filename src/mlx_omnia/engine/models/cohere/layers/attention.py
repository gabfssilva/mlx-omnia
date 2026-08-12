from mlx_omnia.engine.core.attention import HeadLayerNorm, NormalizedFusedQKVAttention
from mlx_omnia.engine.models.cohere.config import CohereConfig


class CohereAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: CohereConfig) -> None:
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        super().__init__(
            config.hidden_size,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            traditional=True,
            norm_eps=config.layer_norm_eps,
            normalize=config.use_qk_norm,
            norm_layout="shaped",
            query_norm=HeadLayerNorm(heads, head_dim, config.layer_norm_eps),
            key_norm=HeadLayerNorm(kv_heads, head_dim, config.layer_norm_eps),
            qkv_bias=config.attention_bias,
            output_bias=config.attention_bias,
        )
        self.queries = heads * head_dim
        self.key_values = kv_heads * head_dim
