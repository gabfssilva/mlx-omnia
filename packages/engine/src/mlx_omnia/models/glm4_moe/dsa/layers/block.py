import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import SwiGLU
from mlx_omnia.core.rope import Yarn, yarn
from mlx_omnia.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from mlx_omnia.models.glm4_moe.dsa.layers.attention import GlmMoEDSAAttention
from mlx_omnia.models.glm4_moe.dsa.layers.cache import DSACache
from mlx_omnia.models.glm4_moe.layers.moe import Glm4MoEMLP


class GlmMoEDSABlock(nn.Module):
    def __init__(self, config: GlmMoEDSAConfig, rope: Yarn, routes: bool) -> None:
        super().__init__()
        self.self_attn = GlmMoEDSAAttention(config, rope)
        self.mlp: Glm4MoEMLP | SwiGLU = (
            Glm4MoEMLP(config.routing())
            if routes
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: DSACache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class GlmMoEDSATrunk(nn.Module):
    def __init__(self, config: GlmMoEDSAConfig) -> None:
        super().__init__()
        rope = yarn(config.qk_rope_head_dim, config.theta, config.rope_scaling)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            GlmMoEDSABlock(config, rope, layer >= config.first_k_dense_replace)
            for layer in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
