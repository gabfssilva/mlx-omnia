from sideros.core.attention import NormalizedFusedQKVAttention
from sideros.core.rope import llama3_freqs
from sideros.models.apertus.config import ApertusConfig


class ApertusAttention(NormalizedFusedQKVAttention):
    def __init__(self, config: ApertusConfig) -> None:
        freqs = (
            None
            if config.scaling is None
            else llama3_freqs(config.head_dim, config.rope_theta, config.scaling)
        )
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            norm_eps=config.rms_norm_eps,
            rope_freqs=freqs,
        )
