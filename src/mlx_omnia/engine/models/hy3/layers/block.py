import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.kernels.add_norm import AddRmsNorm
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.hy3.config import SPARSE, Hy3Config
from mlx_omnia.engine.models.hy3.layers.attention import Hy3Attention
from mlx_omnia.engine.models.hy3.layers.moe import Hy3SparseMoe


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
        self._join_norm: AddRmsNorm | None = None

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVStore
    ) -> mx.array:
        attended, h = self._join(x, mask, cache)
        mlp = self.mlp
        if x.shape[1] == 1 and isinstance(mlp, Hy3SparseMoe) and mlp.fused_step_applies():
            return mlp.fused_step(h, attended)
        return attended + self.mlp(h)

    def _add_norm(self) -> AddRmsNorm:
        """Resolved once, at the first step — after load, when the leaf is final."""
        join = self._join_norm
        if join is None:
            join = AddRmsNorm(self.post_attention_layernorm)
            self._join_norm = join
        return join

    def _join(
        self, x: mx.array, mask: mx.array | str | None, cache: KVStore
    ) -> tuple[mx.array, mx.array]:
        """(x + attention, its post-norm)."""
        attended = self.self_attn(self.input_layernorm(x), cache, mask)
        return self._add_norm()(x, attended)


class Hy3Trunk(nn.Module):
    def __init__(self, config: Hy3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Hy3Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
