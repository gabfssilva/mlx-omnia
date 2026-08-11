from mlx_omnia.core.attention import FusedQKVAttention
from mlx_omnia.models.phi3.config import Phi3Config


class Phi3Attention(FusedQKVAttention):
    def __init__(self, config: Phi3Config) -> None:
        long_rope = config.long_rope
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_dims=config.rope_dims,
            rope_scale=config.rope_scale if long_rope is None else 1.0,
            rope_freqs=None if long_rope is None else long_rope.freqs,
            rope_input_scale=1.0 if long_rope is None else long_rope.scale,
        )
        self.long_rope = long_rope
