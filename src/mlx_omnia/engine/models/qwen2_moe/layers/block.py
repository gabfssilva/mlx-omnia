import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.models.qwen2_moe.config import Qwen2MoEConfig
from mlx_omnia.engine.models.qwen2_moe.layers.attention import Qwen2MoEAttention
from mlx_omnia.engine.models.qwen2_moe.layers.moe import Qwen2MoEMLP


class Qwen2MoEBlock(nn.Module):
    def __init__(self, config: Qwen2MoEConfig) -> None:
        super().__init__()
        self.self_attn = Qwen2MoEAttention(config)
        self.mlp = Qwen2MoEMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVStore) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Qwen2MoETrunk(nn.Module):
    def __init__(self, config: Qwen2MoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2MoEBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
