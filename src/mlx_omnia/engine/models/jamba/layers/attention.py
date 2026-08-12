import mlx.nn as nn

from mlx_omnia.engine.core.attention import SeparateQKVAttention
from mlx_omnia.engine.models.jamba.config import JambaConfig


class JambaAttention(SeparateQKVAttention):
    def __init__(self, config: JambaConfig) -> None:
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden = config.hidden_size
        super().__init__(
            nn.Linear(hidden, heads * head_dim, bias=False),
            nn.Linear(hidden, kv_heads * head_dim, bias=False),
            nn.Linear(hidden, kv_heads * head_dim, bias=False),
            nn.Linear(heads * head_dim, hidden, bias=False),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=1.0,
            rope_dims=0,
        )
