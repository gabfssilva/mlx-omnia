import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.models.gemma4.config import Gemma4TextConfig
from mlx_omnia.engine.models.gemma4.layers.activation import gelu
from mlx_omnia.engine.models.gemma4.layers.attention import Gemma4Attention
from mlx_omnia.engine.models.gemma4.layers.mlp import Gemma4MLP


class Gemma4Block(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.layer_scalar = mx.ones((1,), dtype=mx.float32)
        # PLE leaves live on the block itself: the checkpoint names them
        # layers.N.per_layer_input_gate, not under a nested arm module.
        self.has_ple = config.hidden_size_per_layer_input > 0
        if self.has_ple:
            hidden = config.hidden_size
            ple_dim = config.hidden_size_per_layer_input
            self.per_layer_input_gate = nn.Linear(hidden, ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(ple_dim, hidden, bias=False)
            self.post_per_layer_input_norm = nn.RMSNorm(hidden, eps=eps)

    def __call__(
        self,
        x: mx.array,
        cache: LayerCache,
        per_layer_input: mx.array | None = None,
    ) -> mx.array:
        attended = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cache)
        )
        h = attended + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(attended))
        )
        if self.has_ple:
            assert per_layer_input is not None
            gate = gelu(self.per_layer_input_gate(h))
            projected = self.per_layer_projection(gate * per_layer_input)
            h = h + self.post_per_layer_input_norm(projected)
        return h * self.layer_scalar
