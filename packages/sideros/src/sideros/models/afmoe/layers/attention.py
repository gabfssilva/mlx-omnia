import mlx.nn as nn

from sideros.core.attention import SeparateQKVAttention
from sideros.models.afmoe.config import AfmoeConfig


class AfmoeAttention(SeparateQKVAttention):
    def __init__(self, config: AfmoeConfig, sliding: bool) -> None:
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden = config.hidden_size
        queries = heads * head_dim
        key_values = kv_heads * head_dim
        super().__init__(
            nn.Linear(hidden, queries, bias=False),
            nn.Linear(hidden, key_values, bias=False),
            nn.Linear(hidden, key_values, bias=False),
            nn.Linear(queries, hidden, bias=False),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            rope_dims=head_dim if sliding else 0,
            window=config.sliding_window if sliding else None,
            automatic_mask=True,
            query_norm=nn.RMSNorm(head_dim, eps=config.rms_norm_eps),
            key_norm=nn.RMSNorm(head_dim, eps=config.rms_norm_eps),
            gate=nn.Linear(hidden, queries, bias=False),
        )
        self.queries = queries
        self.rotary = sliding
