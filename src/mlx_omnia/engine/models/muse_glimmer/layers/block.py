import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.core.masks import SLIDING
from mlx_omnia.engine.models.muse_glimmer.config import MuseGlimmerTextConfig
from mlx_omnia.engine.models.muse_glimmer.layers.attention import MuseGlimmerAttention


class MuseGlimmerBlock(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_type: str, rope_theta: float) -> None:
        super().__init__()
        self.self_attn = MuseGlimmerAttention(config, layer_type == SLIDING, rope_theta)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        hidden = config.hidden_size
        self.input_layernorm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=config.post_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(hidden, eps=config.post_norm_eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        attended = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cache))
        return attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )


class MuseGlimmerTrunk(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MuseGlimmerBlock(config, layer_type, rope_theta)
            for layer_type, rope_theta in zip(
                config.layer_types, config.layer_rope_theta, strict=True
            )
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
