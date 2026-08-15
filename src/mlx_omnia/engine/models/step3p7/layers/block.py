import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.models.step3p7.config import Step3p7TextConfig
from mlx_omnia.engine.models.step3p7.layers.attention import Step3p7Attention
from mlx_omnia.engine.models.step3p7.layers.mlp import Step3p7MLP
from mlx_omnia.engine.models.step3p7.layers.moe import Step3p7MoE


class Step3p7Block(nn.Module):
    def __init__(self, config: Step3p7TextConfig, layer: int) -> None:
        super().__init__()
        self.self_attn = Step3p7Attention(config, layer)
        self.routes = layer in config.moe_layers
        if self.routes:
            self.moe = Step3p7MoE(config, layer)
        else:
            self.mlp = Step3p7MLP(
                config.hidden_size,
                config.intermediate_size,
                config.limits[layer],
                config.limits[layer],
            )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVStore
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache, mask)
        h = self.post_attention_layernorm(attended)
        if self.routes:
            return self.moe.step(h, attended)
        return attended + self.mlp(h)


class Step3p7Trunk(nn.Module):
    def __init__(self, config: Step3p7TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Step3p7Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
