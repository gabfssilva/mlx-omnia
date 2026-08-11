from sideros.core.attention import NormalizedFusedQKVAttention
from sideros.core.rope import llama3_freqs
from sideros.models.exaone4.config import Exaone4Config


class Exaone4Attention(NormalizedFusedQKVAttention):
    def __init__(self, config: Exaone4Config, local: bool) -> None:
        rotary = config.rotary or local
        freqs = (
            None
            if config.rope_scaling is None
            else llama3_freqs(config.head_dim, config.rope_theta, config.rope_scaling.llama3)
        )
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_dims=config.head_dim if rotary else 0,
            rope_freqs=freqs,
            norm_eps=config.rms_norm_eps,
            window=config.sliding_window if local else None,
            automatic_mask=True,
        )
        self.rotary = rotary
