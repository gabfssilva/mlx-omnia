import mlx.nn as nn

from mlx_omnia.core.attention import SeparateQKVAttention
from mlx_omnia.models.bitnet.config import BitNetConfig
from mlx_omnia.models.bitnet.layers.bitlinear import BitLinear


class BitNetAttention(SeparateQKVAttention):
    def __init__(self, config: BitNetConfig) -> None:
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.attention_head_dim
        hidden = config.hidden_size
        queries = heads * head_dim
        key_values = kv_heads * head_dim
        super().__init__(
            BitLinear(hidden, queries),
            BitLinear(hidden, key_values),
            BitLinear(hidden, key_values),
            BitLinear(queries, hidden),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            output_norm=nn.RMSNorm(hidden, eps=config.rms_norm_eps),
        )
