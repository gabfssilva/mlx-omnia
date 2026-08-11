import mlx.nn as nn

from mlx_omnia.core.attention import NormalizedFusedQKVAttention
from mlx_omnia.models.olmoe.config import OlmoEConfig


class OlmoEAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: OlmoEConfig) -> None:
        heads = config.num_attention_heads
        kv_heads = config.kv_heads
        head_dim = config.head_dim
        queries = heads * head_dim
        key_values = kv_heads * head_dim
        super().__init__(
            config.hidden_size,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            norm_eps=config.rms_norm_eps,
            norm_layout="flat",
            query_norm=nn.RMSNorm(queries, eps=config.rms_norm_eps),
            key_norm=nn.RMSNorm(key_values, eps=config.rms_norm_eps),
        )
        self.queries = queries
        self.key_values = key_values
