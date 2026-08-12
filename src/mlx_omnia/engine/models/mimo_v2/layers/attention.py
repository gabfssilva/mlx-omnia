import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attention import SeparateQKVAttention
from mlx_omnia.engine.models.mimo_v2.config import SLIDING, LayerType, MimoV2Config


class MimoV2Attention(SeparateQKVAttention):
    def __init__(self, config: MimoV2Config, layer_type: LayerType) -> None:
        sliding = layer_type == SLIDING
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads * (2 if sliding else 1)
        head_dim = config.head_dim
        value_dim = config.v_head_dim
        hidden = config.hidden_size
        bias = config.attention_bias
        super().__init__(
            nn.Linear(hidden, heads * head_dim, bias=bias),
            nn.Linear(hidden, kv_heads * head_dim, bias=bias),
            nn.Linear(hidden, kv_heads * value_dim, bias=bias),
            nn.Linear(heads * value_dim, hidden, bias=False),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            v_head_dim=value_dim,
            rope_theta=config.rope_for(layer_type).rope_theta,
            rope_dims=config.rope_dims(layer_type),
            value_scale=config.value_scale,
            window=config.sliding_window if sliding else None,
            automatic_mask=True,
            sinks=mx.zeros((heads,)) if sliding else None,
        )
