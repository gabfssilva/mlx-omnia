from mlx_omnia.core.attention import NormalizedFusedQKVAttention
from mlx_omnia.core.masks import SLIDING
from mlx_omnia.models.gemma3.config import Gemma3TextConfig


class Gemma3Attention(NormalizedFusedQKVAttention):
    def __init__(self, config: Gemma3TextConfig, layer_type: str) -> None:
        sliding = layer_type == SLIDING
        rope_base = config.rope_local_base_freq if sliding else config.rope_theta
        super().__init__(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=rope_base,
            norm_eps=config.rms_norm_eps,
            attention_scale=config.query_pre_attn_scalar**-0.5,
            window=config.sliding_window if sliding else None,
            automatic_mask=True,
        )
        self.sliding = sliding
        self.rope_base = rope_base
