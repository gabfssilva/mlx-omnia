import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.layers import QuantizedSwitchLinear, SwiGLU
from sideros.models.llama4.config import Llama4TextConfig
from sideros.models.llama4.layers.attention import Llama4Attention
from sideros.models.llama4.layers.moe import Llama4MoE


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

    def _fused_step_applies(self) -> bool:
        mlp = self.mlp
        if not isinstance(mlp, Llama4MoE):
            return False
        gate_up = mlp.switch_mlp.gate_up_proj
        down = mlp.switch_mlp.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                self.hidden, mlp.switch_mlp.inner, gate_up.group_size, down.group_size
            )
        )

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and self._fused_step_applies():
            mlp = self.mlp
            assert isinstance(mlp, Llama4MoE)
            gate_up = mlp.switch_mlp.gate_up_proj
            down = mlp.switch_mlp.down_proj
            assert isinstance(gate_up, QuantizedSwitchLinear)
            assert isinstance(down, QuantizedSwitchLinear)
            assert gate_up.biases is not None and down.biases is not None
            logits = mlp.gate(h).reshape(-1)
            chosen = mx.argmax(logits, axis=-1).astype(mx.uint32)
            indices = mx.reshape(chosen, (1,))
            score = mx.sigmoid(logits[chosen].astype(mx.float32)).astype(h.dtype)
            x_scaled = h.reshape(-1) * score
            act = moe_gate_up_act(
                x_scaled,
                gate_up.weight,
                gate_up.scales,
                gate_up.biases,
                indices,
                group_size=gate_up.group_size,
                bits=gate_up.bits,
            )
            residual = attended.reshape(-1) + mlp.shared_expert(h).reshape(-1)
            return moe_down_combine(
                act.reshape(-1),
                down.weight,
                down.scales,
                down.biases,
                indices,
                mx.array([1.0], dtype=h.dtype),
                residual,
                group_size=down.group_size,
                bits=down.bits,
            ).reshape(1, 1, self.hidden)
        return attended + self.mlp(h)
