import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache, SharedKVReader
from sideros.models.gemma3n.config import Gemma3nTextConfig
from sideros.models.gemma3n.layers.altup import AltUp
from sideros.models.gemma3n.layers.attention import Gemma3nAttention
from sideros.models.gemma3n.layers.laurel import Laurel
from sideros.models.gemma3n.layers.mlp import Gemma3nMLP, gelu


class Gemma3nBlock(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer: int) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.self_attn = Gemma3nAttention(config, layer)
        self.mlp = Gemma3nMLP(config, layer)
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.altup = AltUp(config)
        self.laurel = Laurel(config)
        self.per_layer_input_gate = nn.Linear(
            hidden, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input, hidden, bias=False
        )
        self.post_per_layer_input_norm = nn.RMSNorm(hidden, eps=eps)
        self.active = config.altup_active_idx
        self.correct_scale = config.altup_correct_scale

    def __call__(
        self, x: mx.array, cache: KVCache | SharedKVReader, per_layer_input: mx.array
    ) -> mx.array:
        predictions = self.altup.predict(x)
        active = predictions[self.active]
        normed = self.input_layernorm(active)
        laurel = self.laurel(normed)
        attended = active + self.post_attention_layernorm(self.self_attn(normed, cache))
        joined = (attended + laurel) * (2.0**-0.5)
        mixed = joined + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(joined))
        )
        corrected = self.altup.correct(predictions, mixed)

        first = corrected[self.active]
        if self.correct_scale:
            first = first * self.altup.correct_output_scale
        gated = gelu(self.per_layer_input_gate(first)) * per_layer_input
        projected = self.post_per_layer_input_norm(self.per_layer_projection(gated))
        return mx.concatenate([corrected[:1], corrected[1:] + projected], axis=0)
