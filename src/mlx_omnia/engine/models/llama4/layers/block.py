import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.kernels.down_combine import DownCombine
from mlx_omnia.engine.core.kernels.gate_up import GateUp
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.models.llama4.config import Llama4TextConfig
from mlx_omnia.engine.models.llama4.layers.attention import Llama4Attention
from mlx_omnia.engine.models.llama4.layers.moe import Llama4MoE


class Llama4Block(nn.Module):
    def __init__(self, config: Llama4TextConfig, layer_idx: int, freqs: mx.array) -> None:
        super().__init__()
        self.self_attn = Llama4Attention(config, layer_idx, freqs)
        if layer_idx in config.sparse_layers:
            self.mlp = Llama4MoE(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.dense_intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden = config.hidden_size
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

    def _kernels(self) -> tuple[GateUp, DownCombine] | None:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final. `None` on the dense layers."""
        mlp = self.mlp
        if not isinstance(mlp, Llama4MoE):
            return None
        gate_up, down = self._gate_up, self._down
        if gate_up is None or down is None:
            switch = mlp.switch_mlp
            gate_up = GateUp(switch.gate_up_proj, hidden=self.hidden, inner=switch.inner)
            down = DownCombine(switch.down_proj, hidden=self.hidden, inner=switch.inner)
            self._gate_up, self._down = gate_up, down
        return gate_up, down

    def _fused_step_applies(self) -> bool:
        return isinstance(self.mlp, Llama4MoE)

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: LayerCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and x.shape[0] == 1 and self._fused_step_applies():
            mlp = self.mlp
            assert isinstance(mlp, Llama4MoE)
            kernels = self._kernels()
            assert kernels is not None
            gate_up, down = kernels
            logits = mlp.gate(h).reshape(-1)
            chosen = mx.argmax(logits, axis=-1).astype(mx.uint32)
            indices = mx.reshape(chosen, (1,))
            score = mx.sigmoid(logits[chosen].astype(mx.float32)).astype(h.dtype)
            x_scaled = h.reshape(-1) * score
            act = gate_up(x_scaled, indices)
            residual = attended.reshape(-1) + mlp.shared_expert(h).reshape(-1)
            return down(act, indices, mx.array([1.0], dtype=h.dtype), residual).reshape(
                1, 1, self.hidden
            )
        return attended + self.mlp(h)
