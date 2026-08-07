import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.add_rms_norm import add_rms_norm, add_rms_norm_applies
from sideros.core.layers import SwiGLU
from sideros.models.hy3.config import SPARSE, Hy3Config
from sideros.models.hy3.layers import flags
from sideros.models.hy3.layers.attention import Hy3Attention
from sideros.models.hy3.layers.moe import Hy3SparseMoe


class Hy3Block(nn.Module):
    def __init__(self, config: Hy3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Hy3Attention(config)
        if config.layer_types[layer_idx] == SPARSE:
            self.mlp = Hy3SparseMoe(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden_size
        self.k = config.num_experts_per_tok

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended, h = self._join(x, mask, cache)
        mlp = self.mlp
        if x.shape[1] == 1 and isinstance(mlp, Hy3SparseMoe) and mlp.fused_step_applies():
            return mlp.fused_step(h, attended)
        return attended + self.mlp(h)

    def _join(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> tuple[mx.array, mx.array]:
        """(x + attention, its post-norm). At T=1 one kernel does the pair."""
        if flags.ADD_RMS_NORM_KERNEL and x.shape[1] == 1 and add_rms_norm_applies(self.hidden):
            normed = mx.fast.rms_norm(x, self.input_layernorm.weight, self.eps)
            return add_rms_norm(
                x,
                self.self_attn(normed, mask, cache),
                self.post_attention_layernorm.weight,
                self.eps,
            )
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return attended, self.post_attention_layernorm(attended)


class Hy3Trunk(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Hy3Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
