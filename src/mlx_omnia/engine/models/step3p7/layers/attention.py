import mlx.core as mx

from mlx_omnia.engine.core.attention import (
    GatedNormalizedFusedQKVAttention,
    smooth_rotary_freqs,
)
from mlx_omnia.engine.models.step3p7.config import SLIDING, Step3p7TextConfig


class Step3p7Attention(GatedNormalizedFusedQKVAttention):
    def __init__(self, config: Step3p7TextConfig, layer: int) -> None:
        layer_type = config.types[layer]
        sliding = layer_type == SLIDING
        rotary_dim = int(config.head_dim * config.rotary_factors[layer])
        theta = config.thetas[layer]
        scaling = config.rope_scaling
        freqs = (
            smooth_rotary_freqs(
                rotary_dim,
                theta,
                factor=scaling.factor,
                original_max=scaling.original_max_position_embeddings,
                low_frequency_factor=scaling.low_freq_factor,
                high_frequency_factor=scaling.high_freq_factor,
            )
            if scaling is not None and layer_type in config.yarn_only_types
            else None
        )
        if freqs is not None:
            mx.eval(freqs)
        super().__init__(
            config.hidden_size,
            heads=config.heads_per_layer[layer],
            kv_heads=config.num_attention_groups,
            head_dim=config.head_dim,
            rope_theta=theta,
            rope_dims=rotary_dim,
            rope_freqs=freqs,
            norm_eps=config.rms_norm_eps,
            gate_activation="sigmoid",
            gate_per_head=True,
            window=config.sliding_window if sliding else None,
        )
