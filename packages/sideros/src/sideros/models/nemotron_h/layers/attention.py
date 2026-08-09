import mlx.nn as nn

from sideros.core.attention import SeparateQKVAttention
from sideros.models.nemotron_h.config import NemotronHConfig


class NemotronHAttention(SeparateQKVAttention):
    def __init__(self, config: NemotronHConfig) -> None:
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.attention_head_dim
        hidden = config.hidden_size
        bias = config.attention_bias
        super().__init__(
            nn.Linear(hidden, heads * head_dim, bias=bias),
            nn.Linear(hidden, kv_heads * head_dim, bias=bias),
            nn.Linear(hidden, kv_heads * head_dim, bias=bias),
            nn.Linear(heads * head_dim, hidden, bias=bias),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=1.0,
            rope_dims=0,
        )
