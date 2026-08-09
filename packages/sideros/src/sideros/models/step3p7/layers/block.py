import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import KVCache
from sideros.core.kernels.down_combine import DownCombine
from sideros.core.kernels.gate_up import GateUp
from sideros.core.kernels.moe_route import sigmoid_topk, softmax_topk_applies
from sideros.models.step3p7.config import Step3p7TextConfig
from sideros.models.step3p7.layers.attention import Step3p7Attention
from sideros.models.step3p7.layers.mlp import Step3p7MLP
from sideros.models.step3p7.layers.moe import Step3p7MoE


class Step3p7Block(nn.Module):
    def __init__(self, config: Step3p7TextConfig, layer: int) -> None:
        super().__init__()
        self.self_attn = Step3p7Attention(config, layer)
        if layer in config.moe_layers:
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
        self._gate_up: GateUp | None = None
        self._down: DownCombine | None = None

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache, mask)
        h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and self._fused_step_applies():
            moe = self.moe
            kernels = self._kernels()
            assert kernels is not None
            gate_up, down = kernels
            gate_input = (
                h.reshape(-1).astype(mx.float32)
                if moe.need_fp32_gate
                else h.reshape(-1)
            )
            gate_logits = moe.gate(gate_input)
            bias = (
                moe.router_bias
                if hasattr(moe, "router_bias")
                else mx.zeros(moe.gate.weight.shape[0], mx.float32)
            )
            assert isinstance(bias, mx.array)
            chosen, weights = sigmoid_topk(gate_logits, bias, moe.k, scale=moe.scaling)
            act = gate_up(h.reshape(-1), chosen)
            shared_out = moe.shared_expert(h)
            residual = attended + shared_out
            return down(act, chosen, weights, residual.reshape(-1)).reshape(1, 1, moe.hidden)
        if hasattr(self, "moe"):
            return attended + self.moe(h)
        return attended + self.mlp(h)

    def _kernels(self) -> tuple[GateUp, DownCombine] | None:
        """Resolved once, at the first T=1 step — after load, when the leaves'
        formats are final. `None` on the dense layers."""
        if not hasattr(self, "moe"):
            return None
        moe = self.moe
        gate_up, down = self._gate_up, self._down
        if gate_up is None or down is None:
            switch = moe.switch_mlp
            gate_up = GateUp(switch.gate_up_proj, hidden=moe.hidden, inner=switch.inner)
            down = DownCombine(switch.down_proj, hidden=moe.hidden, inner=switch.inner)
            self._gate_up, self._down = gate_up, down
        return gate_up, down

    def _fused_step_applies(self) -> bool:
        if not hasattr(self, "moe"):
            return False
        moe = self.moe
        return (
            softmax_topk_applies(moe.split + moe.k, moe.k)
            and moe.switch_mlp.gate_limit == 0.0
            and moe.shared_expert.gate_limit == 0.0
        )


class Step3p7Trunk(nn.Module):
    def __init__(self, config: Step3p7TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Step3p7Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
